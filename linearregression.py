from numpy import *


def run():

	#step1 - collect our data

	points=genfromtxt('data.csv',delimiter=',')


	#step2-define our hyperparameters
	#how fast should our model converge?

	learning_rate=0.0001
	
	#y=mx+b(slope formula)

	initial_b=0
	initial_m=0
	num_iteration=1000

	#step3 - train our model

	print 'starting gradient descent at  b ={0},m={1},errpr={2}'.format(initial_b,initial_m,compute_error_for_line_given_points(initial_b,initial_m))

	[b,m]=gradient_descent_runner(points,initial_b,initial_m,learning_rate,num_iteration)


	print 'ending gradient descent at  b ={1},m={2},errpr={3}'.format(num_iteration,b,m,compute_error_for_line_given_points(b,m,points))

if __main__ =='__main__':
	run()