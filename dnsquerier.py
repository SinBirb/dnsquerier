"""Research tool to make DNS queries in different intensities.
Depends on numpy and dnspython"""
import sys
import signal
import time
import argparse
import re
import logging
import asyncio
#import uvloop
#asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

import dns.rdatatype
import dns.resolver
import dns.message

import numpy as np
from itertools import zip_longest

abort = False

def prepareQueries(names):
	""" If binaryMode is true, returns queries in binary representation, otherwise
	returns an array of dns.message.Message objects. IDs are all the same,
	because the objects are references"""
	queries = []
	counter = 0
	# a mapping of string : bytearray
	unique_names = {} 

	for name in names:
		if not (name in unique_names):
			r = dns.renderer.Renderer()
			r.add_question(dns.name.from_text(name), dns.rdatatype.A)
			r.write_header()
			# save as bytearray because later the id needs to be changed
			unique_names[name] = bytearray(r.get_wire())
		queries.append(unique_names[name])
		counter += 1
	return queries


class DNSProtocol:
	def __init__(self, loop, queriesOutstanding, ids, printResponses):
		self.loop = loop
		self.transport = None
		# the waiting list for queries that were sent
		self.queriesOutstanding = queriesOutstanding
		self.ids = ids
		self.printResponses = printResponses

	def connection_made(self, transport):
		self.transport = transport

	def datagram_received(self, data, addr):
		queryId = int.from_bytes(data[0:2], byteorder='big')
		
		if queryId in self.queriesOutstanding:
			log.debug("Received response for id {}".format(queryId))
			if self.printResponses:
				print (dns.message.from_wire(data))

			self.ids[queryId] = True
		else:
			log.error("Received response for id {} which was not sent".
				format(queryId))

	def error_received(self, exc):
		print('Error received:', exc)

	def connection_lost(self, exc):
		print("Socket closed, stop sending")
		self.loop.stop()
	
@asyncio.coroutine
def send_and_wait(transport, message, ids, count):
	global abort
	# needed for parallel sending, should take <1 microsecond
	message_copy = bytearray(message)
	resend = False
	while not ids[count] and not abort:
		if resend:
			log.info("Resending {}".format(count))
		# set id via bit operation
		id_b = count.to_bytes(2, 'big')
		message_copy[0] = id_b[0]
		message_copy[1] = id_b[1]
		
		transport.sendto(message_copy)
		yield from asyncio.sleep(3.0)
		resend = True

@asyncio.coroutine
def send_queries(loop, server, port, distribution, printResponses = True,
	infiniteMode = False):
	"""Send queries according to distribution.
	@param server: string describing the receiver, e.g. "192.168.1.1"
	@param distribution: if infiniteMode is true, a single dns message as
	binary string. Otherwise a list of tuples where each tuple contains a
	bytes object representing a dns query and a number.	The number in the
	tuple defines the pause in miliseconds after sending the query."""
	global abort, log
	# generate some IDs in random order. This is vulnerable to a
	# birthday attack, so it should not be used in real life if
	# the network is not secure. However when ids are shuffled,
	# there is a small probability that two queries with the same
	# id are in the air, resulting in non-uniqueness
	# big endian == network byte order
	#ids = np.arange(65535, dtype=np.dtype('>u2'))
	#np.random.shuffle(ids)
	# needed if ids are shuffled
	#idsRecv = [(ids[x], [False]) for x in range(len(ids))]

	ids = range(65535)
	queriesOutstanding = {}
	# ids of which the queries have been received
	idsRecv = [False] * 65535

	if not infiniteMode:
		log.debug("distribution size: " + str(len(distribution)))

	connect = loop.create_datagram_endpoint(
		lambda: DNSProtocol(loop, queriesOutstanding, idsRecv, printResponses),
		remote_addr=(server, port))
	# wait for endpoint initialization
	transport, protocol = yield from asyncio.ensure_future(connect)

	log.debug("start queries")
	start = time.time()
	count = 0
	for query, pauseTime in distribution:
		if abort:
			break
		# asynchronously send and wait for answer, as soon as possible
		# (if pauseTime is too low this will not enter the function yet)
		queriesOutstanding[count] = asyncio.ensure_future(
			send_and_wait(transport, query, idsRecv, count))

		count = (count + 1) % 65535

		# Provide a chance to run other methods.
		if pauseTime > 0:
			log.debug ("Sent {}, sleeping {}s".format(count, pauseTime))
			yield from asyncio.sleep(pauseTime)
	
	# now wait for all still running background tasks
	log.debug("Finished sending {} queries, reading remaining responses"
		.format(count))
	yield from asyncio.gather(*queriesOutstanding.values())
	
	if abort:
		log.info("Cancelled. Sent {} queries, answers not parsed yet after"
			" {} seconds: {}".format(count, time.time() - start,
				count - idsRecv.count(True)))
	else:
		log.info("Sent and received {} queries after {} seconds"
			.format(count, time.time() - start))

def getRealizationsFromDistribution(queries, distribution, count,
	low=0.0, high=1.0, mean = 0.0, stddev = 1.0, _lambda = 1.0 ):
	""" get count number of float values within low and high as a list,
	using the distribution specified as a string. Used parameters, besides
	size, for the distributions:
	- uniform: low, high
	- normal: mean, stddev (standard deviation)
	- poisson: _lambda"""
	# use thread-safe random generator, because we might use multiple threads
	# https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState
	prng = np.random.RandomState()
	ret = []
	last = queries[len(queries)-1]
	
	if distribution == "uniform":
		tmp = prng.uniform(low, high, count)
		return list(zip_longest(queries, tmp, fillvalue = last))
	elif distribution == "normal":
		tmp = prng.normal(mean, stddev, count)
		tmp[tmp < 0] = 0
		return list(zip_longest(queries, tmp, fillvalue = last))
	elif distribution == "poisson":
		return list(zip_longest(queries, prng.poisson(_lambda, count),
			fillvalue = last))
	else:
		raise Exception("Invalid distribution: {}".format(distribution))

def getSpecialDistribution(queries, kind, burstCount=1, requestsPerBurst=1,
	pauseLength=1.0):
	"""get a distribution that virtualizes some specifique network situation.
	totalTime is the total amount of time the query transmission will take.
	Used parameters for the distributions:
	- bursts: burstCount, requestsPerBurst, pauseLength(pause between bursts)"""
	ret = []
	i = 0
	query = None
	c = 0
	if burstCount < 1 or requestsPerBurst < 1:
		raise Exception("Invalid parameter for bursts mode")
	
	if kind == "bursts":
		for i in range(burstCount):
			for j in range(requestsPerBurst):
				if len(queries) != 0:
					query = queries.pop()
				else:
					c += 1
				if j == requestsPerBurst - 1:
					ret.append( (query, pauseLength) )
				else:
					ret.append( (query, 0) )
		if c > 0:
			log.warning("Filled up with the last name {} times".format(c))
		return ret
	elif kind == "infinite":
		# return a generator
		return loopList([(queries, 0.0001) for query in queries])
	elif kind == "file":
		# TODO: take timestamps from some kind of file
		raise Exception("Not yet implemented")
	else:
		raise Exception("Invalid kind of distribution: {}".format(kind))

def loopList(lst):
	l = len(lst)
	i = 0
	while True:
		yield lst[i]
		i = (i + 1) % l

def main(argv):
	global log
	signal.signal(signal.SIGINT, signal_handler)
	
	logging.basicConfig(format='[dnsquerier %(asctime)s.%(msecs)d %(levelname)s]'
		' %(message)s', datefmt='%H:%M:%S', stream=sys.stdout)
	log = logging.getLogger()
	log.setLevel(logging.WARNING) 
	
	argparser = argparse.ArgumentParser('Send UDP DNS queries in numerous ways')
	argparser.add_argument('-d', '--distribution', default='infinite',
		choices=['uniform', 'normal', 'poisson', 'bursts', 'infinite'],
		help='Distribution for query sending, default: infinite queries'
		+ ', uniform, normal, poisson, bursts. If distribution is infinite, all'
		+ ' names will be sent infinitely in a repeating sequence. Generated'
		+ ' values < 0 for normal distribution are set to 0.')
	argparser.add_argument('-P', '--parameters', default={},
		action=StoreNameValuePair,
		help='Parameters for the distribution as comma separated list or using '
		+ 'the parameter multiple times, e.g. -P count=5,high=2. Possible '
		+ 'options: count (number of requests), low, high, mean, stddev, lambda, '
		+ 'burstCount, requestsPerBurst, pauseLength (seconds between bursts). ')
	argparser.add_argument('-s', '--server', default='127.0.0.1',
		help='Default: 127.0.0.1')
	argparser.add_argument('-p', '--port',   default=53, type=int,
		help='Default: 53')
	argparser.add_argument('-n', '--name', type=str, default='',
		help='DNS name to look up. Only one value allowed')
	argparser.add_argument('-N', '--names-file', type=argparse.FileType('r'),
		help='path to a file that contains the DNS names. Each name has to be '
		+ 'in one line. All names will be held in memory.')
	argparser.add_argument('-l', '--log-responses', default=False,
		action='store_true', help="Wait for responses and print them to stdout")
	argparser.add_argument('-v', '--verbose', default=0, action='count',
		help="Increase verbosity level, maximum effect: 2 times")
	options = argparser.parse_args(argv)
	
	names = []

	if options.verbose == 1:
		log.setLevel(logging.INFO)
	if options.verbose >= 2:
		log.setLevel(logging.DEBUG)
	log.debug(options)
	if options.names_file:
		names = options.names_file.readlines()
		options.names_file.close()
	if options.name != '':
		names.append(options.name)
	if len(names) == 0:
		log.error("At least one name is required")
		sys.exit(1)

	queries = prepareQueries(names)

	loop = asyncio.get_event_loop()
	# TODO: switch off
	loop.set_debug(False)
	
	if options.distribution == "bursts":
		try:
			distribution = getSpecialDistribution(queries, "bursts",
				**options.parameters)
		except TypeError:
			log.exception("Error: Wrong parameter for distribution")
			sys.exit(1)
	elif options.distribution == 'infinite':
		distribution = getSpecialDistribution(queries, "infinite")
		log.info("Queries prepared")
		loop.run_until_complete(send_queries(loop, options.server,
			options.port, distribution, options.log_responses, True))
		return 0
	else:
		if not 'count' in options.parameters:
			options.parameters['count'] = 1
		try:
			distribution = getRealizationsFromDistribution(queries,
				distribution, **options.parameters)
		except TypeError:
			log.exception("Error: Wrong parameter for distribution")
			sys.exit(1)

	log.info("Queries prepared")
	loop.run_until_complete(send_queries(loop, options.server,
		options.port, distribution, options.log_responses))
	loop.close()
	return 0

def signal_handler(signal, frame):
	global abort
	if abort:
		print('\nAnother SIGINT received. Instant exit.')
		sys.exit(1)
	print('\nSIGINT received. Shutting down gracefully.')
	abort = True

class StoreNameValuePair(argparse.Action):
	def __call__(self, parser, namespace, values, option_string=None):
		obj = dict()
		kv_list = [x.split('=') for x in values.split(",")]
		for (key, val) in kv_list:
			if (re.search(r'[^a-zA-Z]', key)):
				raise Exception("Only letters as option key allowed")
			if key == 'lambda':
				key = '_lambda'
			if key in ['low', 'high', 'mean', 'stddev', 'lambda',
				'pauseLength']:
				val = float(val)
			else:
				val = int(val)
			obj[key] = val
			#setattr(namespace, key, val)
		if namespace.parameters:
			namespace.parameters.update(obj)
		else:
			setattr(namespace, "parameters", obj)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

