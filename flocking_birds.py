import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree
import warnings

# İnteraktif mod için (Jupyter/Spyder kullanıyorsanız gerekebilir)
# %matplotlib qt 

warnings.filterwarnings("ignore")

class Octonion3D:
    def __init__(self, neighbor_vels):
        self.v = np.zeros(8)
        # 7 komşunun hızlarını oktoniyon bileşenlerine dağıtarak sentezleme
        for i, v in enumerate(neighbor_vels[:7]):
            # Her komşunun yön bilgisini bir faz olarak kodluyoruz
            self.v[i+1] = np.arctan2(v[1], v[0]) + np.arctan2(v[2], np.sqrt(v[0]**2 + v[1]**2))

    @staticmethod
    def mul(q, r):
        a, b = q[:4], q[4:]; c, d = r[:4], r[4:]
        def q_mul(x, y):
            return np.array([
                x[0]*y[0] - x[1]*y[1] - x[2]*y[2] - x[3]*y[3],
                x[0]*y[1] + x[1]*y[0] + x[2]*y[3] - x[3]*y[2],
                x[0]*y[2] - x[1]*y[3] + x[2]*y[0] + x[3]*y[1],
                x[0]*y[3] + x[1]*y[2] - x[2]*y[1] + x[3]*y[0]])
        def q_conj(x): return np.array([x[0], -x[1], -x[2], -x[3]])
        return np.concatenate([q_mul(a, c) - q_mul(q_conj(d), b), q_mul(d, a) + q_mul(b, q_conj(c))])

    def get_direction(self):
        res = np.zeros(8)
        for i in range(1, 8):
            unit = np.zeros(8); unit[i] = 1.0
            res += self.mul(self.v, unit)
        # Sentezlenen 3D birim vektör
        out = res[1:4]
        mag = np.linalg.norm(out)
        return out / mag if mag > 1e-6 else np.random.randn(3)

class StarlingFlock3D:
    def __init__(self, n=80, size=100):
        self.n, self.size = n, size
        self.pos = np.random.rand(n, 3) * 40 + 30
        self.vel = np.random.randn(n, 3)
        self.vel /= np.linalg.norm(self.vel, axis=1)[:, None]
        self.colors = plt.cm.plasma(np.linspace(0, 1, n))
        self.steer_timer = 0
        self.steer_vec = np.random.randn(3)

    def update(self):
        tree = KDTree(self.pos)
        _, indices = tree.query(self.pos, k=8)
        new_vel = np.zeros_like(self.vel)

        # 1. ANİ YÖN DEĞİŞTİRME (Evrimsel Hedef Şaşırtma)
        self.steer_timer += 1
        if self.steer_timer > 40: 
            self.steer_vec = np.random.randn(3) * 2.5
            self.steer_timer = 0

        for i in range(self.n):
            # Oktoniyonik Hizalanma
            O = Octonion3D(self.vel[indices[i, 1:]])
            alignment = O.get_direction()

            # Topolojik Kohezyon ve Ayrılma
            cohesion = (np.mean(self.pos[indices[i, 1:]], axis=0) - self.pos[i]) * 0.1
            separation = np.zeros(3)
            for idx in indices[i, 1:]:
                diff = self.pos[i] - self.pos[idx]
                d = np.linalg.norm(diff)
                if d < 4.0: separation += diff / (d + 0.1)

            # Sert Küp Sınırı (Köşelere hapsolmayı önleyen itiş)
            avoid = np.zeros(3)
            margin = 12
            for d in range(3):
                if self.pos[i, d] < margin: avoid[d] = 2.0
                elif self.pos[i, d] > self.size - margin: avoid[d] = -2.0

            # Bileşke Steering
            steering = (alignment * 1.5) + (cohesion * 0.5) + (separation * 4.0) + \
                       (avoid * 6.0) + (self.steer_vec * 0.8)

            v_next = self.vel[i] + steering * 0.2
            new_vel[i] = v_next / np.linalg.norm(v_next)

        self.vel = new_vel
        self.pos += self.vel * 2.2

# --- İNTERAKTİF GÖRSELLEŞTİRME ---
sim = StarlingFlock3D(100)
fig = plt.figure(figsize=(10, 8), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')

# İnteraktivite Notu: Pencere açıldığında mouse ile döndürebilirsiniz.
plt.ion() 

def animate(frame):
    sim.update()
    # ax.clear() yerine veriyi güncellemek performansı artırır ancak 3D'de zordur.
    # Bu yüzden temizleyip çiziyoruz ama sınırları sabit tutuyoruz.
    ax.clear()
    ax.set_facecolor('black')
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_zlim(0, 100)
    ax.axis('off')
    
    # Küp iskeletini çiz
    r = [0, 100]
    for s, e in [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]]:
        v = np.array([[r[i//4%2], r[i//2%2], r[i%2]] for i in range(8)])
        ax.plot3D(*zip(v[s], v[e]), color="white", alpha=0.1, lw=0.5)

    # Kuşları Quiver (Ok) ile çiz ki yönleri belli olsun
    ax.quiver(sim.pos[:, 0], sim.pos[:, 1], sim.pos[:, 2],
              sim.vel[:, 0], sim.vel[:, 1], sim.vel[:, 2],
              color=sim.colors, length=4, normalize=True, alpha=0.8)

# Interval'i düşürerek daha akıcı bir interaktivite sağlıyoruz
ani = FuncAnimation(fig, animate, frames=1000, interval=20)
plt.show()
