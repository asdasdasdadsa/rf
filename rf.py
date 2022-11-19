import numpy as np


class Node:
    def __init__(self):
        self.right = None
        self.left = None
        self.split_ind = None
        self.split_val = None
        self.T = None


class DT:
    def __init__(self, max_depth, min_entropy, min_elem, K, M, L_1, L_2):
        self.max_depth = max_depth
        self.min_entropy = min_entropy
        self.min_elem = min_elem
        self.root = Node()
        self.K = K
        self.M = M
        self.L_1 = L_1
        self.L_2 = L_2

    def terminal_node_output(self, cl, dn):
        # def tno(cll, dnn):
        # nn = np.zeros(self.K)
        # for i in range(len(cl)):
        #     nn[np.argmax(cl[i])] += 1
        # return nn / len(cl)
        nn = np.zeros(self.K)
        dd = np.unique(cl, return_counts=True)
        nn[dd[0]] = dd[1]
        return nn / len(cl)

        # return tno(cl, dn)

    def entropy(self, cl, dn):
        # z = np.zeros(self.K)
        # for i in range(len(cl)):
        #     z[np.argmax(cl[i])] += 1
        # z /= len(dn)
        z = np.unique(cl, return_counts=True)[1] / len(dn)
        log_z = np.log(z)
        # return -np.sum(np.array([z[i] * log_z[i] for i in range(self.K)]))
        return - np.sum(np.multiply(z, log_z))

    def information_gain(self, cl, dn, dn_j, cl_j):
        I = self.entropy(cl, dn)
        for i in range(2):
            I -= len(dn_j[i]) * self.entropy(cl_j[i], dn_j[i])
        return I

    def gen_fun(self, cl, dn):
        indexxx = 0
        top = []
        I = np.random.choice(np.arange(64), self.L_1, replace=False)
        for i in I:
            psi = lambda x: x[i]
            J = np.random.uniform(0, 1, self.L_2)
            for j in J:
                # for h in range(len(dn)):
                #     if psi(dn[h]) > j:
                #         dn_left.append(dn[h])
                #         cl_left.append(cl[h])
                #     else:
                #         dn_right.append(dn[h])
                #         cl_right.append(cl[h])

                ind_left = dn[:,i] > j
                ind_right = ~ind_left
                dn_left = dn[ind_left]
                dn_right = dn[ind_right]
                cl_left = cl[ind_left]
                cl_right = cl[ind_right]

                if indexxx %10 == 0:
                    print(indexxx)
                indexxx += 1
                infor = self.information_gain(cl, dn, [dn_left, dn_right], [cl_left, cl_right])
                if len(top) == 0:
                    top.append([psi, infor, [dn_left, dn_right], [cl_left, cl_right], j])
                else:
                    if top[0][1] < infor:
                        top[0] = [psi, infor, [dn_left, dn_right], [cl_left, cl_right], j]
        return top[0]

    def build_tree(self, dn, cl, node, depth):
        entropy_val = self.entropy(cl, dn)
        if depth >= self.max_depth or entropy_val <= self.min_entropy or len(dn) <= self.min_elem:
            node.T = self.terminal_node_output(cl, dn)
        else:
            f = self.gen_fun(cl, dn)
            node.split_ind = f[0]
            node.split_val = f[4]
            dn_left, cl_left = np.array(f[2])[0], np.array(f[3])[0]
            dn_right, cl_right = np.array(f[2])[1], np.array(f[3])[1]

            print(str(entropy_val) + ",,, " + str(depth) + " ,,, ")

            node.left = Node()
            node.right = Node()
            self.build_tree(dn_left, cl_left, node.left, depth + 1)
            self.build_tree(dn_right, cl_right, node.right, depth + 1)

    def pass_tree(self, node, dn):
        if node.T is None:
            if node.split_ind(dn) > node.split_val:
                return self.pass_tree(node.left, dn)
            else:
                return self.pass_tree(node.right, dn)
        else:
            return node.T

    def pass_random_forest(self, node, dn):
        rf = np.zeros(10)
        for i in node:
            dddd = self.pass_tree(i, dn)
            rf += dddd
        return rf/self.M

    def accuracy_tree(self, cl, dn, node):
        err = 0
        for i in range(len(dn)):
            if np.argmax(self.pass_tree(node, dn[i])) == cl[i]: #np.argmax(cl[i]):
                err += 1
        return err / len(dn)

    def accuracy_random_forest(self, cl, dn, node):
        err = 0
        for i in range(len(dn)):
            if np.argmax(self.pass_random_forest(node, dn[i])) == cl[i]: #np.argmax(cl[i]):
                err += 1
        return err / len(dn)
        # for i in node:
        #     err += self.accuracy_tree(cl, dn, i)
        # return err/self.M
