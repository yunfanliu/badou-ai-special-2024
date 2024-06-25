import numpy as np


class BP:
    def __init__(self, arr_input, n_neuron1, true_o,b1,b2, learn):
        w_itoh = np.zeros((len(arr_input), n_neuron1))
        w_htoo = np.zeros((n_neuron1, len(true_o)))

        for i in range(len(arr_input)):
            for j in range(n_neuron1):
                w_itoh[i, j] = np.random.random()

        for i in range(n_neuron1):
            for j in range(len(true_o)):
                w_htoo[i, j] = np.random.random()

        self.arr_input = arr_input  # 输入数据
        self.n_neuron1 = n_neuron1  # 隐藏层神经元数
        self.w_itoh = w_itoh    # 随机输入层-隐藏层权重
        self.w_htoo = w_htoo    # 随机隐藏层-输出层权重
        self.true_o = true_o    # 实际值
        self.b1 = b1
        self.b2 = b2
        self.learn = learn # 学习率

    def net(self, n, w, k, b):
        '''
        n: 输出神经元个数
        w: 权重
        k: 输入
        b: 偏置
        '''
        net = np.zeros(n)
        for i in range(n):
            # print(i)
            net[i] = sum(w[:, i] * k) + b
        return net

    def sigmoid(self, n, s):
        sig = np.zeros(n)
        for i in range(n):
            sig[i] = 1/(1 + np.exp(-s[i]))
        return sig

    def run(self):
        # n 循环次数
        n = 1

        # 初始化
        new_w_itoh = self.w_itoh
        new_w_htoo = self.w_htoo
        zh = self.net(self.n_neuron1, new_w_itoh, self.arr_input, self.b1)
        ah = self.sigmoid(self.n_neuron1, zh)
        zo = self.net(len(self.true_o),  new_w_htoo, ah, self.b2)
        ao = self.sigmoid(len(self.true_o), zo)
        Etotal = sum((self.true_o-ao)**2/len(self.true_o))
            # print(zh)
            # print(ah)
            # print(zo)
            # print(ao)
            # print(Etotal)
            # # return Etotal


        while n != 0:

            partial_derivative_w_htoo = -(self.true_o - ao) * ao *( 1- ao) * ah
            # print(partial_derivative_w_htoo)
            # print(self.w_htoo)
            new_w_htoo = new_w_htoo - self.learn * partial_derivative_w_htoo

            E_ah = np.zeros(len(self.true_o)) # Etotal对ah的求导
            E = np.zeros(len(self.true_o))
            partial_derivative_w_itoh = np.zeros(len(self.arr_input))
            for i in range(len(true_o)):
                    w_htoo = new_w_htoo[i, :]
                    for j in range(len(true_o)):
                        E[j] = -(1/len(self.true_o)) * 2 * (true_o[j] - ao[j]) * ao[j] * (1-ao[j]) * w_htoo[j]
                    E_ah[i] = sum(E)

            for i in range(len(true_o)):
                partial_derivative_w_itoh[i] = E_ah[i] * ah[i] * (1-ah[i]) * self.arr_input[i]
            # print('partial_derivative_w_itoh')
            # print(partial_derivative_w_itoh)
            new_w_itoh = new_w_itoh - self.learn * partial_derivative_w_itoh
            # print(i)
            # print(new_w_itoh)

            #更新新权重后的数据
            zh = self.net(self.n_neuron1,  new_w_itoh, self.arr_input, self.b1)
            ah = self.sigmoid(self.n_neuron1, zh)
            zo = self.net(len(self.true_o), new_w_htoo, ah, self.b2)
            ao = self.sigmoid(len(self.true_o), zo)
            Etotal = sum((self.true_o - ao) ** 2 / len(self.true_o))

            n += 1

            if abs(true_o[0] - ao[0]) < 0.001:
                print(n-1)
                print(ao)
                break



arr_input = np.array((0.05, 0.1))
true0 = 0.01
true_o = np.array((0.01, 1 - true0))
n_neuron1 = 2
b1 = 0.35
b2 = 0.60
learn = 0.5

BP1 = BP(arr_input, n_neuron1, true_o, b1, b2, learn)
a= BP1.run()
print(a)