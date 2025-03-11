#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

/* -----------------------
 * get current time in sec
 * ----------------------- */
double get_time() {
    struct timespec ts[1];
    clock_gettime(CLOCK_REALTIME, ts);
    return ts->tv_sec + ts->tv_nsec * 1.0e-9;
}

// add debug
#ifdef ENABLE_PROFILING
#define BEFORE(name)                                                           \
    double __start_##name = get_time();                                        \
    printf("Start profiling: %s\n", #name);

#define AFTER(name)                                                            \
    double __end_##name = get_time();                                          \
    printf("End profiling: %s, Duration: %.8lf seconds\n", #name,              \
           __end_##name - __start_##name);
#else
#define BEFORE(name)
#define AFTER(name)
#endif

// Define constants
const double PI = 3.14159265358979323846;
const int FRAME_NUM = 100;
const int DIM = 3;
const int NUM_PARTICLES = 10000;
const int MAX_NUM_PARTICLES_PER_CELL = 100;
const double TIME_DELTA = 1.0 / 20.0;
const double ESPILON = 1e-5;
const double PARTICLE_RADIUS = 3.0;
const double PARTICLE_RADIUS_IN_WORLD = PARTICLE_RADIUS / 20.0;
const double GRAVITY[DIM] = {0.0, -10.0, 0.0};
const double BOUNDARY_LOWER = 0.0;
const double BOUNDARY_UPPER = 30.0;
const double GENERATE_LOWER = 5.0;
const double GENERATE_UPPER = 25.0;
const double SPACING =
    (GENERATE_UPPER - GENERATE_LOWER) / std::cbrt(NUM_PARTICLES);
const double INIT_POS[DIM] = {GENERATE_LOWER, BOUNDARY_UPPER - GENERATE_UPPER + GENERATE_LOWER - 0.5, GENERATE_LOWER};
const double CELL_SIZE = 2.51;
const double CELL_RECPROCAL = 1.0 / CELL_SIZE;

// PBF parameters
const double H = 1.1;
const double MASS = 1.0;
const double RHO0 = 1.0;
const double LAMBDA_EPSILON = 100.0;
const int PBF_NUM_ITERS = 5;
const double CORR_DELTAQ_COEFF = 0.3;
const double CORR_K = 0.001;
const double NEIGHBOR_RADIUS = 1.05 * H;
const double DAMPING_FACTOR = 0.01;
const double ELASTICITY = 0.3;

const double POLY6_FACTOR = 315.0 / (64.0 * PI);
const double SPIKY_GRAD_FACTOR = -45.0 / PI;

// Define data structures
struct Particle {
    double old_position[DIM];
    double position[DIM];
    double velocity[DIM];
    double lambda;              // \lambda_i
    double delta_position[DIM]; // position correction
    std::vector<int> neighbors;
};

std::vector<Particle> particles(NUM_PARTICLES);

// For hashing
struct TupleHash {
    std::size_t operator()(const std::tuple<int, int, int> &t) const {
        std::size_t h1 = std::hash<int>()(std::get<0>(t));
        std::size_t h2 = std::hash<int>()(std::get<1>(t));
        std::size_t h3 = std::hash<int>()(std::get<2>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

std::unordered_map<std::tuple<int, int, int>, std::vector<int>, TupleHash> grid;

// Utility to compute grid cell key
std::tuple<int, int, int> get_grid_key(const double *position) {
    int cell[DIM];
    for (int i = 0; i < DIM; ++i) {
        cell[i] = static_cast<int>(std::floor(position[i] * CELL_RECPROCAL));
    }
    return std::make_tuple(cell[0], cell[1], cell[2]);
}

inline void vec_add(double *a, const double *b, double scale = 1.0) {
    for (int i = 0; i < DIM; ++i) {
        a[i] += b[i] * scale;
    }
}

inline void vec_copy(double *dest, const double *src) {
    for (int i = 0; i < DIM; ++i) {
        dest[i] = src[i];
    }
}

// Boundary confine
void confine_to_boundary(double *position, double *velocity = nullptr) {
    for (int i = 0; i < DIM; ++i) {
        if (position[i] < BOUNDARY_LOWER + PARTICLE_RADIUS_IN_WORLD) {
            double overshoot =
                BOUNDARY_LOWER + PARTICLE_RADIUS_IN_WORLD - position[i];
            position[i] = BOUNDARY_LOWER + PARTICLE_RADIUS_IN_WORLD +
                          overshoot * ELASTICITY;
            if (velocity)
                velocity[i] = -velocity[i];
        } else if (position[i] > BOUNDARY_UPPER - PARTICLE_RADIUS_IN_WORLD) {
            double overshoot =
                position[i] - (BOUNDARY_UPPER - PARTICLE_RADIUS_IN_WORLD);
            position[i] = BOUNDARY_UPPER - PARTICLE_RADIUS_IN_WORLD -
                          overshoot * ELASTICITY;
            if (velocity)
                velocity[i] = -velocity[i];
        }
    }
}

double poly6_value(double r, double h) {
    if (r >= h)
        return 0.0;
    double x = (h * h - r * r) / (h * h * h);
    return POLY6_FACTOR * x * x * x;
}

// spiky_gradient(r_vec, h, out_grad): standard Spiky kernel gradient
// out_grad = grad W_spiky(r_vec)
void spiky_gradient(const double *r_vec, double h, double *out_grad) {
    // compute dist
    double dist = 0.0;
    for (int d = 0; d < DIM; ++d) {
        dist += r_vec[d] * r_vec[d];
    }
    dist = std::sqrt(dist);

    if (dist > 0.0 && dist < h) {
        double x = (h - dist) / (h * h * h);
        double g_factor = SPIKY_GRAD_FACTOR * x * x; // factor
        double inv_dist = 1.0 / dist;
        for (int d = 0; d < DIM; ++d) {
            out_grad[d] = r_vec[d] * g_factor * inv_dist;
        }
    } else {
        out_grad[0] = out_grad[1] = out_grad[2] = 0.0;
    }
}

// compute_scorr(r_vec) for "artificial pressure" term
double compute_scorr(const double *r_vec) {
    // scorr_ij = -K * (poly6_value(||r_ij||)/ poly6_value(deltaQ))^4
    // deltaQ = corr_deltaQ_coeff * h
    double dist = 0.0;
    for (int d = 0; d < DIM; ++d) {
        dist += r_vec[d] * r_vec[d];
    }
    dist = std::sqrt(dist);
    if (dist >= H)
        return 0.0;

    double val = poly6_value(dist, H) / poly6_value(CORR_DELTAQ_COEFF * H, H);
    // ^ x^4
    val = val * val;
    val = val * val;
    return -CORR_K * val;
}

void init_particles() {
    int num_per_row = static_cast<int>(std::cbrt(NUM_PARTICLES));
    int num_per_floor = num_per_row * num_per_row;
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        int floor = i / num_per_floor;
        int row = (i % num_per_floor) / num_per_row;
        int col = (i % num_per_floor) % num_per_row;

        particles[i].position[0] = INIT_POS[0] + col * SPACING;
        particles[i].position[1] = INIT_POS[1] + floor * SPACING;
        particles[i].position[2] = INIT_POS[2] + row * SPACING;

        particles[i].velocity[0] = 0.0;
        particles[i].velocity[1] = 0.0;
        particles[i].velocity[2] = 0.0;
    }
}

void update_grid() {
    grid.clear();
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        auto key = get_grid_key(particles[i].position);
        grid[key].push_back(i);
    }
}

void find_neighbors() {
    for (auto &particle : particles) {
        particle.neighbors.clear();
        auto cell_key = get_grid_key(particle.position);

        // Iterate over nearby cells
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    auto neighbor_key = std::make_tuple(
                        std::get<0>(cell_key) + dx, std::get<1>(cell_key) + dy,
                        std::get<2>(cell_key) + dz);

                    if (grid.find(neighbor_key) != grid.end()) {
                        for (int neighbor_idx : grid[neighbor_key]) {
                            double dist_sq = 0.0;
                            for (int k = 0; k < DIM; ++k) {
                                double diff =
                                    particles[neighbor_idx].position[k] -
                                    particle.position[k];
                                dist_sq += diff * diff;
                            }
                            if (dist_sq < NEIGHBOR_RADIUS * NEIGHBOR_RADIUS &&
                                dist_sq > 0.0) {
                                particle.neighbors.push_back(neighbor_idx);
                            }
                        }
                    }
                }
            }
        }
    }
}

void apply_gravity_and_update_positions() {
    for (auto &particle : particles) {
        vec_copy(particle.old_position, particle.position);
        vec_add(particle.velocity, GRAVITY, TIME_DELTA);
        vec_add(particle.position, particle.velocity, TIME_DELTA);
        confine_to_boundary(particle.position, particle.velocity);
    }
}

void after_update() {
    for (auto &particle : particles) {
        confine_to_boundary(particle.position);
        for (int i = 0; i < DIM; ++i) {
            particle.velocity[i] =
                (particle.position[i] - particle.old_position[i]) / TIME_DELTA;
        }
    }
}

void substep() {
    BEFORE(compute_lambdas);
    // Compute lambdas
    for (auto &particle : particles) {
        double density = 0.0;
        double grad_i[DIM] = {0.0, 0.0, 0.0};
        double sum_grad_sqr = 0.0;

        for (int neighbor_idx : particle.neighbors) {
            double r_vec[DIM];
            double dist_sq = 0.0;
            for (int k = 0; k < DIM; ++k) {
                r_vec[k] =
                    particle.position[k] - particles[neighbor_idx].position[k];
                dist_sq += r_vec[k] * r_vec[k];
            }
            double dist = std::sqrt(dist_sq);

            if (dist < H && dist > 0.0) {
                // density constraint
                double poly6 = poly6_value(dist, H);
                density += MASS * poly6;
                // spiky grad
                double g[DIM];
                spiky_gradient(r_vec, H, g);
                // accumulate for grad_i
                for (int k = 0; k < DIM; ++k) {
                    grad_i[k] += g[k];
                    // accumulate neighbor grad^2
                    sum_grad_sqr += g[k] * g[k];
                }
            }
        }
        double constraint = (density / RHO0) - 1.0;

        // Then add grad_i^2
        double grad_i_sq = 0.0;
        for (int k = 0; k < DIM; ++k) {
            grad_i_sq += grad_i[k] * grad_i[k];
        }
        sum_grad_sqr += grad_i_sq;

        // compute lambda
        particle.lambda = -constraint / (sum_grad_sqr + LAMBDA_EPSILON);
    }
    AFTER(compute_lambdas);

    BEFORE(compute_position);
    // Compute position deltas
    for (auto &particle : particles) {
        for (int k = 0; k < DIM; ++k) {
            particle.delta_position[k] = 0.0;
        }

        double lambda_i = particle.lambda;
        for (int neighbor_idx : particle.neighbors) {
            double r_vec[DIM];
            for (int k = 0; k < DIM; ++k) {
                r_vec[k] =
                    particle.position[k] - particles[neighbor_idx].position[k];
            }
            // distance
            double dist_sq = 0.0;
            for (int k = 0; k < DIM; ++k) {
                dist_sq += r_vec[k] * r_vec[k];
            }
            double dist = std::sqrt(dist_sq);

            if (dist < H && dist > 0.0) {
                double lambda_j = particles[neighbor_idx].lambda;
                double scorr_ij = compute_scorr(r_vec);

                // spiky gradient
                double g[DIM];
                spiky_gradient(r_vec, H, g);

                // sum into delta
                for (int k = 0; k < DIM; ++k) {
                    particle.delta_position[k] +=
                        (lambda_i + lambda_j + scorr_ij) * g[k];
                }
            }
        }
        for (int k = 0; k < DIM; ++k) {
            particle.delta_position[k] /= RHO0;
        }
    }
    AFTER(compute_position);

    BEFORE(apply_position);
    // Apply position deltas
    for (auto &particle : particles) {
        vec_add(particle.position, particle.delta_position);
    }
    AFTER(apply_position);

    BEFORE(update_velocity);
    // Update velocity using XSPH
    for (auto &particle : particles) {
        double velocity_update[DIM] = {0.0, 0.0, 0.0};
        for (int neighbor_idx : particle.neighbors) {
            double dist_sq = 0.0;
            double diff[DIM];
            for (int k = 0; k < DIM; ++k) {
                diff[k] =
                    particle.position[k] - particles[neighbor_idx].position[k];
                dist_sq += diff[k] * diff[k];
            }
            double dist = std::sqrt(dist_sq);
            if (dist < H && dist > 0.0) {
                double q = (H - dist) / H;
                double poly6 = POLY6_FACTOR * std::pow(q, 3);
                for (int k = 0; k < DIM; ++k) {
                    velocity_update[k] += (particles[neighbor_idx].velocity[k] -
                                           particle.velocity[k]) *
                                          poly6;
                }
            }
        }
        for (int k = 0; k < DIM; ++k) {
            particle.velocity[k] += DAMPING_FACTOR * velocity_update[k];
        }
    }
    AFTER(update_velocity);
}

void save_to_csv(const std::string &filename, int frame) {
    std::ofstream file(filename);
    file << "x,y,z\n";
    for (const auto &particle : particles) {
        file << particle.position[0] << "," << particle.position[1] << ","
             << particle.position[2] << "\n";
    }
    file.close();
}

int main() {
    // Create output directory
    std::filesystem::create_directory("data");
    BEFORE(total);

    init_particles();

    for (int frame = 0; frame < FRAME_NUM; ++frame) {
        BEFORE(apply_gravity);
        apply_gravity_and_update_positions();
        AFTER(apply_gravity);

        BEFORE(update_grid);
        update_grid();
        AFTER(update_grid);

        BEFORE(find_neighbors);
        find_neighbors();
        AFTER(find_neighbors);

        BEFORE(substep);
        for (int iter = 0; iter < PBF_NUM_ITERS; ++iter) {
            substep();
        }
        AFTER(substep);

        BEFORE(after_update);
        after_update();
        AFTER(after_update);

        // Export
        BEFORE(output);
        char filename[64];
        std::snprintf(filename, sizeof(filename), "data/frame_%03d.csv", frame);
        save_to_csv(filename, frame);
        AFTER(output);
    }

    AFTER(total);
    return 0;
}
