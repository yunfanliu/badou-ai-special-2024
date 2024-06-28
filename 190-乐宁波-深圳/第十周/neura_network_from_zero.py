import scipy.special
import numpy


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 初始化网络，设置节点数
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # 设置学习率
        self.learning_rate = learning_rate

        '''
        初始化权重矩阵：
        weight_input_hidden表示输入层和中间层的权重矩阵
        weight_hidden_output表示隐藏层和输出层的权重矩阵
        '''
        self.weight_input_hidden = numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.weight_hidden_output = numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5

        # 激活函数
        self.activation_function = scipy.special.expit

    def train(self):
        pass

    def query(self, inputs):
        hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.weight_hidden_output, hidden_outputs)
        final_output = self.activation_function(final_inputs)
        print(final_output)
        return final_output


if __name__ == '__main__':
    input_nodes = 4
    hidden_nodes = 5
    output_nodes = 6

    learning_rate = 0.3
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    n.query([1.0, -2.0, 3.0, 4.0])
