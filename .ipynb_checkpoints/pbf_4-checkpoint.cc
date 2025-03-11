#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
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
static const double PI = 3.14159265358979323846;
static const int FRAME_NUM = 100;
static const int DIM = 3;
static const int NUM_PARTICLES = 10000;
static const int MAX_NUM_PARTICLES_PER_CELL = 100;
static const double TIME_DELTA = 1.0 / 20.0;
static const double ESPILON = 1e-5;
static const double PARTICLE_RADIUS = 3.0;
static const double PARTICLE_RADIUS_IN_WORLD = PARTICLE_RADIUS / 20.0;
static const double GRAVITY[DIM] = {0.0, -10.0, 0.0};
static const double BOUNDARY_LOWER = 0.0;
static const double BOUNDARY_UPPER = 30.0;
static const double GENERATE_LOWER = 5.0;
static const double GENERATE_UPPER = 25.0;
static const double SPACING =
    (GENERATE_UPPER - GENERATE_LOWER) / std::cbrt(NUM_PARTICLES);
static const double INIT_POS[DIM] = {
    GENERATE_LOWER, BOUNDARY_UPPER - GENERATE_UPPER + GENERATE_LOWER - 0.5,
    GENERATE_LOWER};
static const double CELL_SIZE = 2.51;
static const double CELL_RECPROCAL = 1.0 / CELL_SIZE;

// PBF parameters
static const double H = 1.1;
static const double MASS = 1.0;
static const double RHO0 = 1.0;
static const double LAMBDA_EPSILON = 100.0;
static const int PBF_NUM_ITERS = 5;
static const double CORR_DELTAQ_COEFF = 0.3;
static const double CORR_K = 0.001;
static const double NEIGHBOR_RADIUS = 1.05 * H;
static const double DAMPING_FACTOR = 0.01;
static const double ELASTICITY = 0.3;

static const double POLY6_FACTOR = 315.0 / (64.0 * PI);
static const double SPIKY_GRAD_FACTOR = -45.0 / PI;

// Define data structures
static std::vector<double> pos_x(NUM_PARTICLES), pos_y(NUM_PARTICLES),
    pos_z(NUM_PARTICLES);

static std::vector<double> vel_x(NUM_PARTICLES), vel_y(NUM_PARTICLES),
    vel_z(NUM_PARTICLES);

static std::vector<double> old_pos_x(NUM_PARTICLES), old_pos_y(NUM_PARTICLES),
    old_pos_z(NUM_PARTICLES);

static std::vector<double> delta_x(NUM_PARTICLES), delta_y(NUM_PARTICLES),
    delta_z(NUM_PARTICLES);

static std::vector<double> lambdas(NUM_PARTICLES);

static std::vector<std::vector<int>> neighbors(NUM_PARTICLES);

static inline int idx(int i, int d) { return i * DIM + d; }

// For hashing
struct TupleHash {
    std::size_t operator()(const std::tuple<int, int, int> &t) const {
        std::size_t h1 = std::hash<int>()(std::get<0>(t));
        std::size_t h2 = std::hash<int>()(std::get<1>(t));
        std::size_t h3 = std::hash<int>()(std::get<2>(t));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

static std::unordered_map<std::tuple<int, int, int>, std::vector<int>,
                          TupleHash>
    grid;

// Utility to compute grid cell key
static inline std::tuple<int, int, int> get_grid_key(double x, double y,
                                                     double z) {
    int cx = static_cast<int>(std::floor(x * CELL_RECPROCAL));
    int cy = static_cast<int>(std::floor(y * CELL_RECPROCAL));
    int cz = static_cast<int>(std::floor(z * CELL_RECPROCAL));
    return std::make_tuple(cx, cy, cz);
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
inline void confine_to_boundary(double &x, double &y, double &z,
                                double *vx = nullptr, double *vy = nullptr,
                                double *vz = nullptr) {
    // x
    if (x < BOUNDARY_LOWER + PARTICLE_RADIUS_IN_WORLD) {
        double overshoot = BOUNDARY_LOWER + PARTICLE_RADIUS_IN_WORLD - x;
        x = BOUNDARY_LOWER + PARTICLE_RADIUS_IN_WORLD + overshoot * ELASTICITY;
        if (vx)
            *vx = -(*vx);
    } else if (x > BOUNDARY_UPPER - PARTICLE_RADIUS_IN_WORLD) {
        double overshoot = x - (BOUNDARY_UPPER - PARTICLE_RADIUS_IN_WORLD);
        x = BOUNDARY_UPPER - PARTICLE_RADIUS_IN_WORLD - overshoot * ELASTICITY;
        if (vx)
            *vx = -(*vx);
    }

    // y
    if (y < BOUNDARY_LOWER + PARTICLE_RADIUS_IN_WORLD) {
        double overshoot = BOUNDARY_LOWER + PARTICLE_RADIUS_IN_WORLD - y;
        y = BOUNDARY_LOWER + PARTICLE_RADIUS_IN_WORLD + overshoot * ELASTICITY;
        if (vy)
            *vy = -(*vy);
    } else if (y > BOUNDARY_UPPER - PARTICLE_RADIUS_IN_WORLD) {
        double overshoot = y - (BOUNDARY_UPPER - PARTICLE_RADIUS_IN_WORLD);
        y = BOUNDARY_UPPER - PARTICLE_RADIUS_IN_WORLD - overshoot * ELASTICITY;
        if (vy)
            *vy = -(*vy);
    }

    // z
    if (z < BOUNDARY_LOWER + PARTICLE_RADIUS_IN_WORLD) {
        double overshoot = BOUNDARY_LOWER + PARTICLE_RADIUS_IN_WORLD - z;
        z = BOUNDARY_LOWER + PARTICLE_RADIUS_IN_WORLD + overshoot * ELASTICITY;
        if (vz)
            *vz = -(*vz);
    } else if (z > BOUNDARY_UPPER - PARTICLE_RADIUS_IN_WORLD) {
        double overshoot = z - (BOUNDARY_UPPER - PARTICLE_RADIUS_IN_WORLD);
        z = BOUNDARY_UPPER - PARTICLE_RADIUS_IN_WORLD - overshoot * ELASTICITY;
        if (vz)
            *vz = -(*vz);
    }
}

static inline double poly6_value(double r, double h) {
    if (r >= h)
        return 0.0;
    double hh = h * h;
    double x = (hh - r * r) / (hh * h);
    return POLY6_FACTOR * x * x * x;
}

static inline void spiky_gradient(double rx, double ry, double rz, double dist,
                                  double h, double &gx, double &gy,
                                  double &gz) {
    if (dist > 0.0 && dist < h) {
        double x = (h - dist) / (h * h * h);
        double g_factor = SPIKY_GRAD_FACTOR * x * x;
        double inv_dist = 1.0 / dist;
        gx = rx * g_factor * inv_dist;
        gy = ry * g_factor * inv_dist;
        gz = rz * g_factor * inv_dist;
    } else {
        gx = gy = gz = 0.0;
    }
}

static inline double compute_scorr(double dist, double h) {
    if (dist >= h)
        return 0.0;
    double val = poly6_value(dist, h) / poly6_value(CORR_DELTAQ_COEFF * h, h);
    // val^4
    val = val * val;
    val = val * val;
    return -CORR_K * val;
}

static void init_particles() {
    int num_per_row = static_cast<int>(std::cbrt(NUM_PARTICLES));
    int num_per_floor = num_per_row * num_per_row;
#pragma omp parallel for simd
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        int floor = i / num_per_floor;
        int row = (i % num_per_floor) / num_per_row;
        int col = (i % num_per_floor) % num_per_row;

        pos_x[i] = INIT_POS[0] + col * SPACING;
        pos_y[i] = INIT_POS[1] + floor * SPACING;
        pos_z[i] = INIT_POS[2] + row * SPACING;
        neighbors[i].reserve(64);
        vel_x[i] = vel_y[i] = vel_z[i] = 0.0;
    }
}

static void update_grid() {
    grid.clear();
    int grid_size = static_cast<int>(
        std::ceil(BOUNDARY_UPPER - BOUNDARY_LOWER) * CELL_RECPROCAL);
    grid.reserve(grid_size * grid_size * grid_size);

#pragma omp parallel
    {
        std::unordered_map<std::tuple<int, int, int>, std::vector<int>,
                           TupleHash>
            localGrid;
        localGrid.reserve(grid_size / omp_get_num_threads() + 1);

#pragma omp for nowait
        for (int i = 0; i < NUM_PARTICLES; ++i) {
            auto key = get_grid_key(pos_x[i], pos_y[i], pos_z[i]);
            localGrid[key].push_back(i);
        }

#pragma omp critical
        {
            for (auto &kv : localGrid) {
                auto &global_vec = grid[kv.first];
                global_vec.insert(global_vec.end(), kv.second.begin(),
                                  kv.second.end());
            }
        }
    }
}

static void find_neighbors() {
#pragma omp parallel for
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        neighbors[i].clear();
        auto cell_key = get_grid_key(pos_x[i], pos_y[i], pos_z[i]);
        // Iterate over nearby cells
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    auto neighbor_key = std::make_tuple(
                        std::get<0>(cell_key) + dx, std::get<1>(cell_key) + dy,
                        std::get<2>(cell_key) + dz);

                    auto it = grid.find(neighbor_key);
                    if (it == grid.end())
                        continue;
                    const auto &candidates = it->second;
                    for (int nidx : candidates) {
                        double dx_ = pos_x[nidx] - pos_x[i];
                        double dy_ = pos_y[nidx] - pos_y[i];
                        double dz_ = pos_z[nidx] - pos_z[i];
                        double dist_sq = dx_ * dx_ + dy_ * dy_ + dz_ * dz_;
                        if (dist_sq < (NEIGHBOR_RADIUS * NEIGHBOR_RADIUS) &&
                            dist_sq > 0.0) {
                            neighbors[i].push_back(nidx);
                        }
                    }
                }
            }
        }
    }
}

static void apply_gravity_and_update_positions() {
#pragma omp parallel for simd
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        old_pos_x[i] = pos_x[i];
        old_pos_y[i] = pos_y[i];
        old_pos_z[i] = pos_z[i];

        // v = v + g * dt
        vel_x[i] += GRAVITY[0] * TIME_DELTA;
        vel_y[i] += GRAVITY[1] * TIME_DELTA;
        vel_z[i] += GRAVITY[2] * TIME_DELTA;
    }
#pragma omp parallel for simd
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        // x = x + v * dt
        pos_x[i] += vel_x[i] * TIME_DELTA;
        pos_y[i] += vel_y[i] * TIME_DELTA;
        pos_z[i] += vel_z[i] * TIME_DELTA;
    }
#pragma omp parallel for simd
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        confine_to_boundary(pos_x[i], pos_y[i], pos_z[i], &vel_x[i], &vel_y[i],
                            &vel_z[i]);
    }
}

static void substep() {
    BEFORE(compute_lambdas);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        double density = 0.0;
        double sum_grad_sqr = 0.0;
        double gx_i = 0.0, gy_i = 0.0, gz_i = 0.0;

        double pos_x_i = pos_x[i], pos_y_i = pos_y[i], pos_z_i = pos_z[i];

        for (int nidx : neighbors[i]) {
            double rx = pos_x_i - pos_x[nidx];
            double ry = pos_y_i - pos_y[nidx];
            double rz = pos_z_i - pos_z[nidx];
            double dist_sq = rx * rx + ry * ry + rz * rz;

            if (dist_sq < H * H && dist_sq > ESPILON) {
                double dist = std::sqrt(dist_sq);
                double poly6 = poly6_value(dist, H);
                density += MASS * poly6;

                double gx, gy, gz;
                spiky_gradient(rx, ry, rz, dist, H, gx, gy, gz);
                gx_i += gx;
                gy_i += gy;
                gz_i += gz;
                sum_grad_sqr += (gx * gx + gy * gy + gz * gz);
            }
        }
        double constraint = (density / RHO0) - 1.0;
        sum_grad_sqr += (gx_i * gx_i + gy_i * gy_i + gz_i * gz_i);

        lambdas[i] = -constraint / (sum_grad_sqr + LAMBDA_EPSILON);
    }
    AFTER(compute_lambdas);

    BEFORE(compute_position);
#pragma omp parallel for simd schedule(dynamic)
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        double lambda_i = lambdas[i];
        double dx_ = 0.0, dy_ = 0.0, dz_ = 0.0;

        for (int nidx : neighbors[i]) {
            double rx = pos_x[i] - pos_x[nidx];
            double ry = pos_y[i] - pos_y[nidx];
            double rz = pos_z[i] - pos_z[nidx];
            double dist_sq = rx * rx + ry * ry + rz * rz;

            if (dist_sq < H * H && dist_sq > ESPILON) {
                double dist = std::sqrt(dist_sq);
                double lambda_j = lambdas[nidx];
                double scorr_ij = compute_scorr(dist, H);

                double gx, gy, gz;
                spiky_gradient(rx, ry, rz, dist, H, gx, gy, gz);

                double coeff = (lambda_i + lambda_j + scorr_ij);
                dx_ += coeff * gx;
                dy_ += coeff * gy;
                dz_ += coeff * gz;
            }
        }
        delta_x[i] = dx_ / RHO0;
        delta_y[i] = dy_ / RHO0;
        delta_z[i] = dz_ / RHO0;
    }
    AFTER(compute_position);

    BEFORE(apply_position);
#pragma omp parallel for simd
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        pos_x[i] += delta_x[i];
        pos_y[i] += delta_y[i];
        pos_z[i] += delta_z[i];
    }
    AFTER(apply_position);

    BEFORE(update_velocity);
#pragma omp parallel for simd
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        double vx_update = 0.0, vy_update = 0.0, vz_update = 0.0;

        for (int nidx : neighbors[i]) {
            double rx = pos_x[i] - pos_x[nidx];
            double ry = pos_y[i] - pos_y[nidx];
            double rz = pos_z[i] - pos_z[nidx];
            double dist_sq = rx * rx + ry * ry + rz * rz;

            if (dist_sq < H * H && dist_sq > ESPILON) {
                double q = (H - sqrt(dist_sq)) / H;
                double poly6 = POLY6_FACTOR * q * q * q;

                vx_update += (vel_x[nidx] - vel_x[i]) * poly6;
                vy_update += (vel_y[nidx] - vel_y[i]) * poly6;
                vz_update += (vel_z[nidx] - vel_z[i]) * poly6;
            }
        }
        vel_x[i] += DAMPING_FACTOR * vx_update;
        vel_y[i] += DAMPING_FACTOR * vy_update;
        vel_z[i] += DAMPING_FACTOR * vz_update;
    }
    AFTER(update_velocity);
}

static void after_update() {
#pragma omp parallel for simd
    for (int i = 0; i < NUM_PARTICLES; ++i) {
        confine_to_boundary(pos_x[i], pos_y[i], pos_z[i]);

        vel_x[i] = (pos_x[i] - old_pos_x[i]) / TIME_DELTA;
        vel_y[i] = (pos_y[i] - old_pos_y[i]) / TIME_DELTA;
        vel_z[i] = (pos_z[i] - old_pos_z[i]) / TIME_DELTA;
    }
}

static void save_to_csv(const std::string &filename) {
    const int nThreads = omp_get_max_threads();
    constexpr int MaxLenPerLine = 64;
    std::vector<std::vector<char>> partialBuffers(nThreads);

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int chunkSize = (NUM_PARTICLES + nThreads - 1) / nThreads;
        int start = tid * chunkSize;
        int end = std::min(start + chunkSize, NUM_PARTICLES);
        partialBuffers[tid].reserve((end - start) * MaxLenPerLine);
        char lineBuf[MaxLenPerLine];
        for (int i = start; i < end; i++) {
            char *ptr = lineBuf;
            {
                auto r = std::to_chars(ptr, ptr + MaxLenPerLine, pos_x[i],
                                       std::chars_format::fixed, 6);
                ptr = r.ptr;
            }
            *ptr++ = ',';

            // 写第二个坐标
            {
                auto r = std::to_chars(ptr, ptr + MaxLenPerLine, pos_y[i],
                                       std::chars_format::fixed, 6);
                ptr = r.ptr;
            }
            *ptr++ = ',';

            // 写第三个坐标
            {
                auto r = std::to_chars(ptr, ptr + MaxLenPerLine, pos_z[i],
                                       std::chars_format::fixed, 6);
                ptr = r.ptr;
            }
            *ptr++ = '\n';
            size_t len = size_t(ptr - lineBuf);
            partialBuffers[tid].insert(partialBuffers[tid].end(), lineBuf,
                                       lineBuf + len);
        }
    }
    std::ofstream file(filename, std::ios::binary);
    file << "x,y,z\n";
    for (int t = 0; t < nThreads; ++t) {
        auto &buf = partialBuffers[t];
        file.write(buf.data(), std::streamsize(buf.size()));
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
        save_to_csv(filename);
        AFTER(output);
    }

    AFTER(total);
    return 0;
}
