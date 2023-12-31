import random
import numpy as np
from functools import reduce


def sigmoid(x):
    return 1.0 / (1 + np.exp(x))


class Node(object):
    def __init__(self, layer_index, node_index) -> None:
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        self.output = output

    def append_downstream_connection(self, connection):
        self.downstream.append(connection)

    def append_upstream_connection(self, connection):
        self.upstream.append(connection)

    def calc_output(self):
        output = reduce(lambda sum, connection: sum +
                        connection.upstream_node.output*connection.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda sum, connection: sum+connection.downstream_node.delta*connection.weight, self.downstream, 0.0)
        self.delta = self.output*(1-self.output)*downstream_delta

    def calc_output_layer_delta(self, label):
        self.delta = self.output*(1-self.output)*(label-self.output)

    def __str__(self):
        node_str = '{}-{}:output: {} delta: {}'.format(
            self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(
            lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret +
                              '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


class ConstNode(object):
    # 为了实现一个输出恒为1的节点(计算偏置$w_{b}$项时需要)
    def __init__(self, layer_index, node_index) -> None:
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, connection):
        self.downstream.append(connection)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(
            lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


class Layer(object):
    def __init__(self, layer_index, node_count) -> None:
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        map_to_list_data = list(data)
        for i in range(len(map_to_list_data)):
            self.nodes[i].set_output(map_to_list_data[i])

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print(node)


class Connection(object):
    # 主要职责是记录连接的权重，以及这个连接所关联的上下游节点
    def __init__(self, upstream_node: Node, downstream_node: Node) -> None:
        '''
        初始化连接，权重初始化为是一个很小的随机数
        upstream_node: 连接的上游节点
        downstream_node: 连接的下游节点
        '''
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate*self.gradient

    def calc_gradient(self):
        self.gradient = self.gradient  # self.gradient TODO

    def get_gradient(self):
        return self.gradient

    def __str__(self):
        # return '({}-{}) -> ({}-{}) = {}'.format(
        #     self.upstream_node.layer_index,
        #     self.upstream_node.node_index,
        #     self.downstream_node.layer_index,
        #     self.downstream_node.node_index,
        #     self.weight)
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)


class Connections(object):
    # manipulate the connection
    def __init__(self) -> None:
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for connection in self.connections:
            print(connection)


class Network(object):
    def __init__(self, layers):
        # eg: layers:[8,3,8]
        self.connections = Connections()
        self.layers: list[Layer] = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        # Important: 上游有wb,下游不连wb（输出恒为1的偏置项）
        # 所以一个是nodes，一个是nodes[:,-1]
        # 这是因为偏置项用来加权算下一层的神经元，但是下一层的偏置项却不需要算这一层神经元
        for layer in range(layer_count-1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer+1].nodes[:-1]]
            for connection in connections:
                self.connections.add_connection(connection)
                connection.downstream_node.append_upstream_connection(
                    connection)
                connection.upstream_node.append_downstream_connection(
                    connection)

    def train(self, labels, data_set, rate, epoch):
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        map_to_list_label = list(label)
        output_nodes: list[Node] = self.layers[-1].nodes
        for i in range(len(map_to_list_label)):
            output_nodes[i].calc_output_layer_delta(map_to_list_label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for connection in node.downstream:
                    connection.update_weight(rate)

    def calc_gradient(self):
        '''
        内部函数，计算每个连接的梯度
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        '''
        获得网络在一个样本下，每个连接上的梯度
        label: 样本标签
        sample: 样本输入
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        '''
        根据输入的样本预测输出值
        sample: 数组，样本的特征，也就是网络的输入向量
        '''
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def dump(self):
        for layer in self.layers:
            layer.dump()


class Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        return map(lambda m: 0.9 if number & m else 0.1, self.mask)

    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x, y: x + y, binary)


def mean_square_error(vec1, vec2):
    return 0.5 * reduce(lambda a, b: a + b,
                        map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                            zip(vec1, vec2)
                            )
                        )


def gradient_check(network, sample_feature, sample_label):
    '''
    梯度检查
    network: 神经网络对象
    sample_feature: 样本的特征
    sample_label: 样本的标签
    '''
    # 计算网络误差
    def network_error(vec1, vec2): return \
        0.5 * reduce(lambda a, b: a + b,
                     map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                         zip(vec1, vec2)))

    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)

    # 对每个权重做梯度检查
    for conn in network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()

        # 增加一个很小的值，计算网络的误差
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon  # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)

        # 根据式6计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)

        # 打印
        print('expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient))


def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256, 8):
        n = normalizer.norm(int(random.uniform(0, 256)))
        data_set.append(n)
        labels.append(n)
    return labels, data_set


def train(network):
    labels, data_set = train_data_set()
    network.train(labels, data_set, 0.3, 50)


def test(network, data):
    normalizer = Normalizer()
    norm_data = normalizer.norm(data)
    predict_data = network.predict(norm_data)
    print('\ttestdata(%u)\tpredict(%u)' % (
        data, normalizer.denorm(predict_data)))


def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))


def gradient_check_test():
    net = Network([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)


if __name__ == '__main__':
    net = Network([8, 3, 8])
    train(net)
    net.dump()
    correct_ratio(net)
