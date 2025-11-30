import numpy as np
import matplotlib.pyplot as plt
import copy

# ==========================================
# PART 1: TSP PROBLEM MODELING
# ==========================================
class TSPSystem:
    def __init__(self, num_cities=20, map_size=100):
        self.num_cities = num_cities
        self.coords = np.random.rand(num_cities, 2) * map_size
        self.dist_matrix = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                self.dist_matrix[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])

    def get_tour_from_key(self, key_vector):
        return np.argsort(key_vector)

    def calculate_distance(self, tour):
        dist = 0
        for i in range(len(tour) - 1):
            dist += self.dist_matrix[tour[i], tour[i+1]]
        dist += self.dist_matrix[tour[-1], tour[0]]
        return dist

    def evaluate(self, population):
        scores = []
        for ind in population:
            tour = self.get_tour_from_key(ind)
            scores.append(self.calculate_distance(tour))
        return np.array(scores)

# ==========================================
# PART 2: DE ALGORITHM & VARIANTS
# ==========================================
class DE_Optimizer:
    def __init__(self, system, variant='rand1', NP=50, G_max=200):
        self.system = system
        self.variant = variant
        self.NP = NP
        self.G_max = G_max
        self.D = system.num_cities 
        
        self.pop = np.random.rand(self.NP, self.D)
        self.fitness = self.system.evaluate(self.pop)
        
        self.best_idx = np.argmin(self.fitness)
        self.global_best_fit = self.fitness[self.best_idx]
        self.global_best_ind = self.pop[self.best_idx].copy()
        self.history = []

        # SHADE/JADE Memory & Params
        self.archive = []
        if variant == 'shade':
            self.mem_size = 50 
            self.M_F = np.ones(self.mem_size) * 0.5
            self.M_CR = np.ones(self.mem_size) * 0.5
            self.k_mem = 0

        self.mu_F = 0.5
        self.mu_CR = 0.5

    def optimize(self):
        for g in range(self.G_max):
            new_pop = np.zeros_like(self.pop)
            
            if self.variant in ['jade', 'shade']:
                CR = np.random.normal(self.mu_CR, 0.1, self.NP)
                CR = np.clip(CR, 0, 1)
                F = np.random.standard_cauchy(self.NP) * 0.1 + self.mu_F
                F = np.clip(F, 0, 1)
            else:
                F = np.full(self.NP, 0.5)
                CR = np.full(self.NP, 0.8)

            for i in range(self.NP):
                idxs = [idx for idx in range(self.NP) if idx != i]
                r = self.pop[np.random.choice(idxs, 3, replace=False)]
                
                if self.variant == 'rand1':
                    v = r[0] + F[i] * (r[1] - r[2])
                elif self.variant == 'best1':
                    best = self.pop[self.best_idx]
                    v = best + F[i] * (r[0] - r[1])
                elif self.variant in ['jade', 'shade']:
                    p_best_idx = np.argsort(self.fitness)[:max(1, int(0.05 * self.NP))]
                    p_best = self.pop[np.random.choice(p_best_idx)]
                    pop_union = np.vstack((self.pop, np.array(self.archive))) if len(self.archive) > 0 else self.pop
                    r2 = pop_union[np.random.randint(len(pop_union))]
                    v = self.pop[i] + F[i] * (p_best - self.pop[i]) + F[i] * (r[0] - r2)
                else: # curr2best
                    best = self.pop[self.best_idx]
                    v = self.pop[i] + F[i] * (best - self.pop[i]) + F[i] * (r[0] - r[1])

                cross_points = np.random.rand(self.D) < CR[i]
                if not np.any(cross_points): cross_points[np.random.randint(0, self.D)] = True
                u = np.where(cross_points, v, self.pop[i])
                new_pop[i] = u

            new_fitness = self.system.evaluate(new_pop)
            succ_F, succ_CR, diff_fit = [], [], []

            for i in range(self.NP):
                if new_fitness[i] < self.fitness[i]: 
                    if self.variant in ['jade', 'shade']:
                        self.archive.append(self.pop[i])
                        succ_F.append(F[i])
                        succ_CR.append(CR[i])
                        diff_fit.append(self.fitness[i] - new_fitness[i])
                    
                    self.fitness[i] = new_fitness[i]
                    self.pop[i] = new_pop[i]
                    
                    if new_fitness[i] < self.global_best_fit:
                        self.global_best_fit = new_fitness[i]
                        self.best_idx = i
                        self.global_best_ind = self.pop[i].copy()

            if len(self.archive) > self.NP:
                import random
                random.shuffle(self.archive)
                self.archive = self.archive[:self.NP]
            
            if self.variant == 'shade' and len(succ_F) > 0:
                weights = np.array(diff_fit) / np.sum(diff_fit)
                mean_scr = np.sum(weights * np.array(succ_CR))
                mean_sf = np.sum(weights * np.array(succ_F)**2) / np.sum(weights * np.array(succ_F))
                
                self.M_CR[self.k_mem] = mean_scr
                self.M_F[self.k_mem] = mean_sf
                self.k_mem = (self.k_mem + 1) % self.mem_size
                
                r_idx = np.random.randint(0, self.mem_size)
                self.mu_F = self.M_F[r_idx]
                self.mu_CR = self.M_CR[r_idx]

            self.history.append(self.global_best_fit)
            
        return self.history, self.global_best_ind

# ==========================================
# PART 3: EXECUTION & PLOTTING (ENGLISH)
# ==========================================
if __name__ == "__main__":
    print("Initializing TSP (20 cities)...")
    np.random.seed(42) # Keep seed 42 to match the previous image exactly
    tsp_model = TSPSystem(num_cities=20, map_size=100)
    
    variants = {
        "DE/rand/1": "rand1",
        "DE/best/1": "best1", 
        "DE/curr-to-best": "curr2best",
        "JADE": "jade",
        "SHADE": "shade"
    }
    
    results = {}
    best_route_overall = None
    min_dist_overall = float('inf')

    plt.figure(figsize=(14, 6))

    # PLOT 1: CONVERGENCE
    plt.subplot(1, 2, 1)
    for name, code in variants.items():
        print(f"--- Running {name} ---")
        opt = DE_Optimizer(tsp_model, variant=code, NP=50, G_max=200)
        history, best_ind = opt.optimize()
        
        final_dist = history[-1]
        plt.plot(history, label=f"{name} ({final_dist:.1f})", linewidth=2)
        
        if final_dist < min_dist_overall:
            min_dist_overall = final_dist
            best_route_overall = tsp_model.get_tour_from_key(best_ind)

    # --- ENGLISH LABELS HERE ---
    plt.title("Convergence of DE on TSP (Lower is better)")
    plt.xlabel("Generation")
    plt.ylabel("Total Distance")
    plt.legend()
    plt.grid(True)

    # PLOT 2: BEST ROUTE
    plt.subplot(1, 2, 2)
    coords = tsp_model.coords
    best_tour = best_route_overall
    
    # Draw Cities
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=50, zorder=2)
    for i, (x, y) in enumerate(coords):
        plt.text(x, y+1, str(i), fontsize=9)
    
    # Draw Paths
    for i in range(len(best_tour) - 1):
        p1 = coords[best_tour[i]]
        p2 = coords[best_tour[i+1]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', alpha=0.7, zorder=1)
    
    # Return to start
    p1 = coords[best_tour[-1]]
    p2 = coords[best_tour[0]]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', alpha=0.7, zorder=1)

    # --- ENGLISH TITLE HERE ---
    plt.title(f"Best Route Found (Dist: {min_dist_overall:.1f})")
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig("tsp_de_result.png")
    plt.show()
    print("\nDone! Check file 'tsp_de_result.png' for the English version.")