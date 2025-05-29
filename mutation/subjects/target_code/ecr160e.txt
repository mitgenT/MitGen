from collections import deque


class Dinic:
    def __init__(self, N):
        self.N = N
        self.G = [[] for i in range(N)]

    def add_edge(self, fr, to, cap):
        forward = [to, cap, None]
        forward[2] = backward = [fr, 0, forward]
        self.G[fr].append(forward)
        self.G[to].append(backward)

    def add_multi_edge(self, v1, v2, cap1, cap2):
        edge1 = [v2, cap1, None]
        edge1[2] = edge2 = [v1, cap2, edge1]
        self.G[v1].append(edge1)
        self.G[v2].append(edge2)

    def bfs(self, s, t):
        self.level = level = [None]*self.N
        deq = deque([s])
        level[s] = 0
        G = self.G
        while deq:
            v = deq.popleft()
            lv = level[v] + 1
            for w, cap, _ in G[v]:
                if cap and level[w] is None:
                    level[w] = lv
                    deq.append(w)
        return level[t] is not None

    def dfs(self, v, t, f):
        if v == t:
            return f
        level = self.level
        for e in self.it[v]:
            w, cap, rev = e
            if cap and level[v] < level[w]:
                d = self.dfs(w, t, min(f, cap))
                if d:
                    e[1] -= d
                    rev[1] += d
                    return d
        return 0

    def flow(self, s, t):
        flow = 0
        INF = 10**9 + 7
        G = self.G
        while self.bfs(s, t):
            *self.it, = map(iter, self.G)
            f = INF
            while f:
                f = self.dfs(s, t, INF)
                flow += f
        return flow


H, W = map(int, input().split())
N = H + W
start, goal = N, N + 1
N += 2
X = [list(map(int, input().split())) for _ in range(H)]
A = list(map(int, input().split()))
B = list(map(int, input().split()))
mx_flow = Dinic(N)
for i in range(H):
    for j in range(W):
        v, w = i, j + H
        mx_flow.add_edge(v, w, 1)
for i, a in enumerate(A):
    v, w, c = start, i, a
    mx_flow.add_edge(v, w, c)
for j, b in enumerate(B):
    v, w, c = j + H, goal, b
    mx_flow.add_edge(v, w, c)
f = mx_flow.flow(start, goal)
if not (f == sum(A) and sum(A) == sum(B)):
    print(-1)
    exit()
# print(f)




def add_edge(v, w, cap, cost):
    g[v].append([w, cap, cost, len(g[w])])
    g[w].append([v, 0, -cost, len(g[v]) - 1])


g = [[] for _ in range(N)]
for i in range(H):
    for j in range(W):
        v, w = i, j + H
        if X[i][j]:
            add_edge(v, w, 1, 0)
        else:
            add_edge(v, w, 1, 1)
for i, a in enumerate(A):
    v, w, c = start, i, a
    add_edge(v, w, c, 0)
for j, b in enumerate(B):
    v, w, c = j + H, goal, b
    add_edge(v, w, c, 0)

inf = 10 ** 18


def min_cost_flow(s, t, f):
    res = 0
    prevv = [0] * N
    preve = [0] * N
    while f:
        dist = [inf] * N
        dist[s] = 0
        update = True
        while update:
            update = False
            for v in range(N):
                if dist[v] == inf:
                    continue
                for i, (w, cap, cost, j) in enumerate(g[v]):
                    if cap > 0 and dist[w] > dist[v] + cost:
                        dist[w] = dist[v] + cost
                        prevv[w] = v
                        preve[w] = i
                        update = True
        if dist[t] == inf:
            return -1
        d = f
        v = t
        while v != s:
            d = min(d, g[prevv[v]][preve[v]][1])
            v = prevv[v]
        f -= d
        res += d * dist[t]
        v = t
        while v != s:
            w, cap, cost, j = g[prevv[v]][preve[v]]
            g[prevv[v]][preve[v]][1] -= d
            g[v][j][1] += d
            v = prevv[v]
    return res


min_cost_flow(start, goal, f)
Y = [[0] * W for _ in range(H)]
for v in range(H):
    for w, cap, cost, j in g[v]:
        if 0 <= v < H and 0 <= w - H < W and not cap:
            Y[v][w - H] = 1
ans = 0
for i in range(H):
    for j in range(H):
        if X[i][j] != Y[i][j]:
            ans += 1
    # print(*Y[i])
print(ans)
# for i in range(N):
#     print(g[i])