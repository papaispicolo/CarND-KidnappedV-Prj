/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position
	// (based on estimates of x, y, theta and their uncertainties from GPS)
	// and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Set number of particles to 300
	num_particles = 300;

	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	// TODO: Set standard deviations for x, y, and theta
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(x, std_x);

	// TODO: Create normal distributions for y and theta
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// initialize 300 random particles sampled from
	for (int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(p.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];

	// random noise distributions of x, y, std_theta
	normal_distribution<double> noise_x(0.0, std_x);
	normal_distribution<double> noise_y(0.0, std_y);
	normal_distribution<double> noise_theta(0.0, std_theta);

	// predict particle's next position x, y and std_theta
	for (int i=0; i < num_particles; i++ ) {
		Particle& p = particles[i];
		// from the lesson 12 : Motion Models - 4. Yaw Rate and Velocity
		// if Yaw rate Theta dot = 0
		// x_f = x_0 + v (dt) cos(theta_0)
		// y_f = x_0 + v (dt) sin(theta_0)
		// theta_f = theta_0
		if (fabs(yaw_rate) < 0.0001) {
			p.x = p.x + velocity * delta_t * cos(p.theta);
			p.y = p.y + velocity * delta_t * cos(p.theta);
			// theta <- unchanged
		}
		// from the lesson 12 : Motion Models - 4. Yaw Rate and Velocity
		// if Yaw rate =!= 0
		// x_f = x_0 + v/yaw_rate * ( sin(theta_0 + yaw_rate * dt) - sin (theta_0) )
		// y_f = y_0 + v/yaw_rate * ( cos(theta_0) - cos(theta_0 + yaw_rate * dt ) )
		// theta_f = theta_0 + yaw_rate * dt
		else {
			p.x = p.x + velocity/yaw_rate * ( sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y = p.y + velocity/yaw_rate * ( cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
			p.theta = p.theta + yaw_rate * delta_t;
		}

		// now add random noise
		p.x += noise_x(gen);
		p.y += noise_y(gen);
		p.theta += noise_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); i++) {
		double min_dist = -1.0;
		double l2_dist;

		for (int j = 0; j < predicted.size(); j++) {
			l2_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			if ( l2_dist < min_dist || min_dist < 0) {
				min_dist = l2_dist;
				// update the most closest predicted index j for observation i
				observations[i].id = j;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double l2_dist;
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];

	for ( int i=0; i < num_particles; i++) {
		Particle &p = particles[i];

		// 1. collect the landmarks in given sensor range
		vector<LandmarkObs> landmarks_in_range;

		Map::single_landmark_s lm;
		for ( int j=0; j < map_landmarks.landmark_list.size() ; j++ ) {
			lm = map_landmarks.landmark_list[j];
			// calculate distance between given particle and landmark
			l2_dist = dist(p.x,p.y, lm.x_f,lm.y_f);

			if ( l2_dist < sensor_range ) {
				LandmarkObs lm_obs;
				lm_obs.id = lm.id_i; lm_obs.x = lm.x_f; lm_obs.y = lm.y_f;
				landmarks_in_range.push_back(lm_obs);
			}
		}

		// 2. observation (measurement) cordinate to map coordinate
		vector<LandmarkObs> measurement_in_map;

		for (int j=0; j < observations.size(); j++) {
			LandmarkObs o = observations[j];
			// Lesson 14 : 16. Landmark - Homogenous Transformation
			// Xm = Xp + ( cos(theta) * Xc ) - ( sin(theta) * Yc )
			// Ym = yp + ( sin(theta) * Xc ) + ( cos(theta) * Yc )
			LandmarkObs m_map; // observation / measurement in map coordinate
			m_map.x = p.x + cos(p.theta) * o.x - sin(p.theta) * o.y;
			m_map.y = p.y + sin(p.theta) * o.x + cos(p.theta) * o.y;
			measurement_in_map.push_back(m_map);
		}

		// 3. associate landmarks_in_range and observations_in_map
		// after this step. each observatiosn_in_map is set with nearest landmark index
		dataAssociation(landmarks_in_range, measurement_in_map);

		// 4. find particle associations and update its weights
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;

		p.weight = 1.0 ;// reset weights

		LandmarkObs measurement, landmark;
		for (int j=0; j < measurement_in_map.size(); j++ ) {
			// for all observation and its closest landmark
			measurement = measurement_in_map[j];
			landmark = landmarks_in_range[measurement.id];

			// update associations
			associations.push_back(landmark.id);
			sense_x.push_back(measurement.x);
			sense_y.push_back(measurement.y);

			// Update the particle's weight using a Multivariate Gaussian
			double coeff = 1 / (2 * M_PI * std_x * std_y);
			double power1 = (pow(measurement.x - landmark.x, 2) / (2 * std_x*std_x));
			double power2 = (pow(measurement.y - landmark.y, 2) / (2 * std_y*std_y));

			p.weight *= coeff * exp(-(power1 + power2));
		}

		// Set particle's associations & corresponding parameters
		SetAssociations(p, associations, sense_x, sense_y);
		weights[i]= p.weight;

	}

}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine generator;
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	// Collect samples into new vector
	vector<Particle> particles_samp;
	for(int i=0;i<num_particles;i++){
		particles_samp.push_back(particles[distribution(generator)]);
	}

	// Overwrite existing vector
	particles = particles_samp;

}


void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
