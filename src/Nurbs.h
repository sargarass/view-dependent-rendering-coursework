#pragma once
#include "bothInclude.h"
#include "BezierPatch.h"

struct NURBS {
    int degree_u;
    int degree_v;
    int order_u;
    int order_v;
    int n;
    int m;

    std::vector<std::vector<glm::vec4>> points;
    std::vector<float> u_knots;
    std::vector<float> v_knots;
    NURBS() {
        degree_u = degree_v = 0;
        order_u = order_v = 0;
    }


    void decomposeSurfeceV(int p, int q, float *V, int &nb, std::vector<std::vector<glm::vec4>> &Pw, std::vector<std::vector<std::vector<glm::vec4>>> &Qw) {
        int a = q;
        int b = q + 1;
        nb = 0;
        for (int i = 0; i < p; i++) {
            for (int row = 0; row <= q; row++) {
                Qw[nb][i][row] = Pw[i][row];
            }
        }
        int run = m;
        while (b < run) {
            int i = b;
            while (b < (m + q) && V[b + 1] == V[b]) {
                b++;
            }
            int mult = b - i + 1;
            if (mult < q) {
                double numer = V[b] - V[a];
                double alphas[q];
                for (int j = q; j > mult; j--) {
                    alphas[j - mult - 1] = numer / (V[a + j] - V[a]);
                }

                int r = q - mult;
                for (int j = 1; j <= r; j++) {
                    int save = r - j;
                    int s = mult + j;
                    for (int k = q; k >= s; k--) {
                        float alpha = alphas[k - s];
                        for (int row = 0; row < p; row++) {
                            Qw[nb][row][k] = alpha * Qw[nb][row][k]
                                             + (1.0f - alpha) * Qw[nb][row][k - 1];
                        }
                    }

                    if (b < run) {
                        for (int row = 0; row < p; row++) {
                            Qw[nb + 1][row][save] = Qw[nb][row][q];
                        }
                    }

                }
            }
            nb = nb + 1;
            if (b < run) {
                for (int i = q - mult; i <= q; i++) {
                    for (int row = 0; row < p; row++) {
                        Qw[nb][row][i] = Pw[row][b - q + i];
                    }
                }
                a = b;
                b = b + 1;
            }
        }
    }

    void decomposeSurfaceU(int p, float *U, int &nb, const std::vector<std::vector<glm::vec4>> &Pw, std::vector<std::vector<std::vector<glm::vec4>>> &Qw) {
        int a = p;
        int b = p + 1;
        nb = 0;
        for (int i = 0; i <= p; i++) {
            for (int row = 0; row < m; row++) {
                Qw[nb][i][row] = Pw[i][row];
            }
        }
        int run = n;
        while (b < run) {
            int i = b;
            while (b < (n + p) && U[b + 1] == U[b]) {
                b++;
            }
            int mult = b - i + 1;
            if (mult < p) {
                float numer = U[b] - U[a];
                float alphas[p];
                for (int j = p; j > mult; j--) {
                    alphas[j - mult - 1] = numer / (U[a + j] - U[a]);
                }

                int r = p - mult;
                //****//
                for (int j = 1; j <= r; j++) {
                    int save = r - j;
                    int s = mult + j;
                    for (int k = p; k >= s; k--) {
                        float alpha = alphas[k - s];
                        for (int row = 0; row < m; row++) {
                            Qw[nb][k][row] = alpha * Qw[nb][k][row]
                                             + (1.0f - alpha) * Qw[nb][k - 1][row];
                        }
                    }

                    if (b < run) {
                        for (int row = 0; row < m; row++) {
                            Qw[nb + 1][save][row] = Qw[nb][p][row];
                        }
                    }

                }
            }
            nb = nb + 1;
            if (b < run) {
                for (int i = p - mult; i <= p; i++) {
                    for (int row = 0; row < m; row++) {
                        Qw[nb][i][row] = Pw[b - p + i][row];
                    }
                }
                a = b;
                b = b + 1;
            }
        }
    }

    void decomposeSurface(BezierPatch *out) {
        float *U = &u_knots[0];
        float *V = &v_knots[0];
        std::vector<std::vector<std::vector<glm::vec4>>> Qw;
        Qw.resize(n - order_u + 1);
        for (int i = 0; i < n - order_u + 1; i++) {
            Qw[i].resize(order_u);
            for (int j = 0; j < order_u; j++) {
                Qw[i][j].resize(m);
            }
        }
        int nb = 0;
        decomposeSurfaceU(order_u - 1, U, nb, points, Qw);

        std::vector<std::vector<std::vector<glm::vec4>>> patches(m - order_v + 1);
        for (int i = 0; i < m - order_v + 1; i++) {
            patches[i].resize(order_u);
            for (int j = 0; j < order_u; j++) {
                patches[i][j].resize(order_v);
            }
        }

        size_t it = 0;
        for (int i = 0; i < n - order_u + 1; i++) {
            int num = 0;
            decomposeSurfeceV(order_u, order_v - 1, V, num, Qw[i], patches);
            for (int j = 0; j < m - order_v + 1; j++) {
                for (int v = 0; v < order_v; v++) {
                    for (int u = 0; u < order_u; u++) {
                        out[it].points[v][u] = patches[j][v][u];
                    }
                }
                it++;
            }
        }
    }

    uint64_t getBezierPatchesCount() const {
        return (n - order_u + 1) * (m - order_v + 1);
    }
};
