#@title
import numpy as np
from typing import List
import tensornetwork as tn
from itertools import product

class TNQCircuit:
    """TNQCircuit电路的实现."""

    def __init__(self, num_qubits, backend='jax'):
        self.num_qubits = num_qubits
        self.backend = backend
        # 包含张量网络所需的所有节点的最终列表
        self.network = [self.get_initial_state()]
        # 包含某个特定量子比特应用何种门信息的列表
        self.gate_patch = []
        # control_gate，包含一对   控制量子位和目标量子位的列表
        self.control_gates_patch = []
        # 包含旋转门角度的列表
        self.arguments = [None] * self.num_qubits
        self.graphics_terminal = []
        for index in range(self.num_qubits):
            self.gate_patch.append('I')
            self.graphics_terminal.append("    |  ")
            self.graphics_terminal.append("q%2s |──" % str(index))
            self.graphics_terminal.append("    |  ")

    # 定义一个量子比特门dictionary
    gates = {
        "I": np.eye(2, dtype=np.complex128),

        "X": np.array([[0.0, 1.0],
                       [1.0, 0.0]], dtype=np.complex128),

        "Y": np.array([[0.0, 0.0 - 1.j],
                       [0. + 1.j, 0.0]], dtype=np.complex128),

        "Z": np.array([[1.0, 0.0],
                       [0.0, -1.0]], dtype=np.complex128),

        "H": np.array([[1, 1],
                       [1, -1]], dtype=np.complex128) / np.sqrt(2),

        "T": np.array([[1.0, 0.0],
                       [0.0, np.exp(1.j * np.pi / 4)]], dtype=np.complex128),

        "R": np.array([[1.0, 0.0],
                       [0.0, 1.0]], dtype=np.complex128),

        "RX": np.array([[1.0, 1.0],
                        [1.0, 1.0]], dtype=np.complex128),

        "RY": np.array([[1.0, 1.0],
                        [1.0, 1.0]], dtype=np.complex128),

        "RZ": np.array([[1.0, 0.0],
                        [0.0, 1.0]], dtype=np.complex128)
    }

    ########################################## 图形功能 ############################################

    # 颜色定义
    colors = {
        0: "\u001b[0m",     # 重置
        1: "\u001b[31m",    # 红色
        2: "\u001b[32m",    # 绿色
        3: "\u001b[33m",    # 黄色
        4: "\u001b[34m",    # 蓝色
        5: "\u001b[35m",    # 洋红色
        6: "\u001b[36m",    # 青色
        7: "\u001b[37m",    # 白色
        8: "\u001b[31m",
        9: "\u001b[32m",
        10: "\u001b[33m",
        11: "\u001b[34m",
    }
    # 定义图形函数
    def apply_graphics_to_patch(self):
        """
          可视化所有门应用在其上的电路。
        """
        color_iterator = 1
        for control_gate in self.control_gates_patch:
            full_list = control_gate[0] + [control_gate[1]]
            for qubit in full_list:
                if any(qubit - it_qubit == 1 for it_qubit in full_list):
                    self.graphics_terminal[qubit * 3] += "%s╔═╩═╗%s   " % (self.colors[color_iterator], self.colors[0])
                else:
                    self.graphics_terminal[qubit * 3] += "%s╔═══╗%s   " % (self.colors[color_iterator], self.colors[0])
                if (qubit == control_gate[1]):
                    self.graphics_terminal[qubit * 3 + 1] += "%s║ %s ║%s───" % (
                    self.colors[color_iterator], self.gate_patch[qubit][1], self.colors[0])
                else:
                    self.graphics_terminal[qubit * 3 + 1] += "%s║ %s ║%s───" % (
                    self.colors[color_iterator], self.gate_patch[qubit][1].lower(), self.colors[0])
                if any(qubit - it_qubit == -1 for it_qubit in full_list):
                    self.graphics_terminal[qubit * 3 + 2] += "%s╚═╦═╝%s   " % (
                    self.colors[color_iterator], self.colors[0])
                else:
                    self.graphics_terminal[qubit * 3 + 2] += "%s╚═══╝%s   " % (
                    self.colors[color_iterator], self.colors[0])
            color_iterator = color_iterator + 1

        for qubit in range(self.num_qubits):
            if (self.gate_patch[qubit] == 'I'):
                self.graphics_terminal[qubit * 3] += "        "
                self.graphics_terminal[qubit * 3 + 1] += "────────"
                self.graphics_terminal[qubit * 3 + 2] += "        "
            elif ("Target" not in self.gate_patch[qubit] and "Control" not in self.gate_patch[qubit]):
                if len(self.gate_patch[qubit]) > 1:
                    self.graphics_terminal[qubit * 3] += "╔═══╗   "
                    self.graphics_terminal[qubit * 3 + 1] += "║ %s ║───" % self.gate_patch[qubit]
                    self.graphics_terminal[qubit * 3 + 2] += "╚═══╝   "
                else:
                    self.graphics_terminal[qubit * 3] += "╔═══╗   "
                    self.graphics_terminal[qubit * 3 + 1] += "║ %s ║───" % self.gate_patch[qubit]
                    self.graphics_terminal[qubit * 3 + 2] += "╚═══╝   "

    ########################################## 图形功能 ############################################

    # TNQCircuit类的方法

    def get_initial_state(self):
        """"
          生成并返回量子电路初始状态的节点
        """

        if self.num_qubits <= 0 or not isinstance(self.num_qubits, int):
            raise ValueError("Amount of qubits should be not-negative integer.")
        # 创建初始状态向量
        initial_state = np.zeros(2 ** self.num_qubits, dtype=np.complex128)
        initial_state[0] = 1.0 + 0.j
        initial_state = np.transpose(initial_state)
        # 把初始状态的张量打包到一个节点
        initial_state_node = tn.Node(initial_state, backend=self.backend)
        return initial_state_node

    # 生成两个量子比特门
    def generate_control_gate(self, control, target: List, gate: str):
        """
          考虑给定的控制和目标量子位，为任何具有不同数量的量子位的系统生成并返回控制门的张量。
          参数的意义:
            control: 控制量子位的index
            target: 目标量子位的index
            gate:  我将要生成的一中门 (Xgate, Ygate, etc.)
          返回的是:
            A tensor of the contol gate
        """
        control_gate = np.eye(2 ** self.num_qubits, dtype=np.complex128)
        tuples = []
        # 搜索最多 2**self.num_qubits 的所有数字，
        # 以便在二进制表示中它们在控制位置具有“1”，在目标位置具有“0”。
        for i in range(2 ** self.num_qubits):
            if not (i & (1 << target)) and all(i & (1 << control_qubit) for control_qubit in control):
                swap = i + 2 ** target
                # 将变换嵌入到矩阵中
                control_gate[i][i] = self.gates[gate][0][0]
                control_gate[i][swap] = self.gates[gate][0][1]
                control_gate[swap][i] = self.gates[gate][1][0]
                control_gate[swap][swap] = self.gates[gate][1][1]
        # 如果控制门应用Hadamard门，则将整个系统置于叠加状态
        if gate == 'H':
            control_gate = control_gate * (1. + 1.j) / np.sqrt(2)
        return control_gate

    def apply_arguments(self, gate):
        """
            在量子态上应用R, RX, RY, RZ门
            Args:
              gate: a qubit应用了的gate的数量
            Returns:
              None.
        """
        if self.gate_patch[gate] == 'R':
            self.gates['R'][1][1] = self.arguments[gate]
        if self.gate_patch[gate] == 'RX':
            self.gates['RX'][0][0] = np.cos(self.arguments[gate])
            self.gates['RX'][1][1] = np.cos(self.arguments[gate])
            self.gates['RX'][1][0] = -1.j * np.sin(self.arguments[gate])
            self.gates['RX'][0][1] = -1.j * np.sin(self.arguments[gate])
        if self.gate_patch[gate] == 'RY':
            self.gates['RY'][0][0] = np.cos(self.arguments[gate])
            self.gates['RY'][1][1] = np.cos(self.arguments[gate])
            self.gates['RY'][1][0] = np.sin(self.arguments[gate])
            self.gates['RY'][0][1] = -np.sin(self.arguments[gate])
        if self.gate_patch[gate] == 'RZ':
            self.gates['RZ'][0][0] = np.exp(-1.j * self.arguments[gate])
            self.gates['RZ'][1][1] = np.exp(1.j * self.arguments[gate])

    def evaluate_patch(self):
        """
          评估在给定时刻应用于电路的门
          存储在张量网络节点中的相应门的张量
        """
        if all(self.gate_patch[i] == 'I' for i in range(self.num_qubits)):
            return

        # 调用图形函数
        self.apply_graphics_to_patch()

        # 为当前patch中的所有控制门创建矩阵
        for control_gate_info in self.control_gates_patch:
            target_qubit = control_gate_info[1]
            if self.gate_patch[target_qubit][1] == 'R':
                self.gates['R'][1][1] = self.arguments[target_qubit]
            control_gate = self.generate_control_gate(control_gate_info[0], target_qubit,
                                                      self.gate_patch[target_qubit][1])
            control_gate = control_gate.transpose()
            self.network.append(tn.Node(control_gate, backend=self.backend))
            self.gate_patch[target_qubit] = 'I'
            for qubit in control_gate_info[0]:
                self.gate_patch[qubit] = 'I'
        self.control_gates_patch = []

        self.apply_arguments(self.num_qubits - 1)
        result_matrix = self.gates[self.gate_patch[self.num_qubits - 1]]

        # 用tensor product展开空间
        shape = 4
        for gate in reversed(range(self.num_qubits - 1)):
            self.apply_arguments(gate)
            result_matrix = np.tensordot(result_matrix, self.gates[self.gate_patch[gate]], axes=0)
            result_matrix = result_matrix.transpose((0, 2, 1, 3)).reshape((shape, shape))
            shape = len(result_matrix) * 2
        result_matrix = result_matrix.transpose()
        # 将the moment存储在节点中并附加到电路中
        self.network.append(tn.Node(result_matrix, backend=self.backend))
        for index in range(self.num_qubits):
            self.gate_patch[index] = 'I'
            self.arguments[index] = None

    def get_state_vector(self):
        """
          将 resulting state vector作为rank-1 的张量返回
          将值四舍五入到小数点后 3 位
        """
        # 连接所有节点并评估存储在其中的所有张量
        self.evaluate_patch()

        if len(self.network) > 1:
            for index in reversed(range(1, len(self.network) - 1)):
                self.network[index + 1][0] ^ self.network[index][1]
            self.network[1][0] ^ self.network[0][0]
        nodes = tn.reachable(self.network[1])
        result = tn.contractors.greedy(nodes, ignore_edge_order=True)
        # 将结果四舍五入到小数点后 3 位
        state_vecor = np.round(result.tensor, 3)
        return state_vecor

    # 获取振幅(amplitude)
    def get_amplitude(self):
        """
          打印电路最终状态向量的振幅
          振幅定义为布洛赫球上状态向量的长度
          将结果四舍五入到小数点后 3 位
        """

        state_vector = self.get_state_vector()
        # amplitude = sqrt( (real_part)^2 + (complex_part)^2)
        for index in range(2 ** self.num_qubits):
            amplitude = np.absolute(state_vector[index])
            # 十进制转二进制decimal to binary
            b = np.binary_repr(index, width=self.num_qubits)
            print("|" + b + "> amplitude " + str(amplitude))

    # 获取位串Get bitstring
    def get_bitstring(self):
        """
          打印电路最终状态向量的位串
          Probability calculated 作为 a value 乘以 value_conjugate
          Returns:
            每个比特的概率(Probability)
            最可能的bitstring的二进制表示
        """

        state_vector = self.get_state_vector()
        sample = {}
        # probability = complex_magnitude * complex_magnitude_conjugate
        for index in range(2 ** self.num_qubits):
            probability = state_vector[index] * np.conjugate(state_vector[index])
            probability = np.round(np.real(probability), 3)
            b = np.binary_repr(index, width=self.num_qubits)
            sample[index] = probability
            # print("|" + b + "> probability " + str(probability))
        return sample, np.binary_repr(max(sample, key=sample.get), width=self.num_qubits)

    # 获取可视化
    def visualize(self):
        """
          可视化量子电路
        """
        self.evaluate_patch()
        for string in self.graphics_terminal:
            print(string)

    # 检查正确的输入
    def check_input_one_gate(self, target: int):
        """"
          检查单量子比特门的基本输入
          Args:
            target: 一个目标量子位
          Return: None.
          Raise:
            Value Errors.
        """
        if target > self.num_qubits - 1:
            raise ValueError("Qubit's index exceed the specified size of the cirquit.")
        if target < 0 or not isinstance(target, int):
            raise ValueError("Target gate should be not-negative integer.")

    # 添加量子比特门
    def X(self, target: int):
        """添加 X gate (logical NOT) 到 the stack of current moment.
          Args:
            target: X 门作用于其上的量子比特节点的索引
          Returns: None.
          Raise: ValueError 如果目标 quibit 的 index 超出电路大小
                 ValueError 如果目标 quibit 是浮点数或负数
        """

        self.check_input_one_gate(target)

        # 如果 gates 应用于所有量子比特，则评估current moment并开始填写next moment
        if (self.gate_patch[target] != 'I'):
            self.evaluate_patch()
        self.gate_patch[target] = 'X'

    def Y(self, target: int):
        self.check_input_one_gate(target)

        if (self.gate_patch[target] != 'I'):
            self.evaluate_patch()
        self.gate_patch[target] = 'Y'

    def Z(self, target: int):
        self.check_input_one_gate(target)

        if (self.gate_patch[target] != 'I'):
            self.evaluate_patch()
        self.gate_patch[target] = 'Z'

    def H(self, target: int):
        """
          Hadamard Gate 将初始状态向量带入其叠加状态
        """
        self.check_input_one_gate(target)

        if (self.gate_patch[target] != 'I'):
            self.evaluate_patch()
        self.gate_patch[target] = 'H'

    def T(self, target: int):
        self.check_input_one_gate(target)

        if (self.gate_patch[target] != 'I'):
            self.evaluate_patch()
        self.gate_patch[target] = 'T'

    def R(self, phi: float, target: int):
        """Add R gate to the stack of current moment.
          Args:
            phi: 以弧度为单位的角度，对应于量子位状态围绕 z 轴的给定 phi 值的旋转
        """
        self.check_input_one_gate(target)

        if (self.gate_patch[target] != 'I'):
            self.evaluate_patch()
        # 存储传递的角度值
        self.arguments[target] = np.exp(1.j * phi)
        self.gate_patch[target] = 'R'

    # = = = = = = = = = == = = = = = = = = = == = = = = = = = = = = = = = = = = = = == = =

    def RX(self, phi: float, target: int):
        """Add RX gate to the stack of current moment.
           Args:
              phi: 以弧度表示的角度，对应于量子位状态围绕 X 轴的给定 phi 值的旋转
        """

        self.check_input_one_gate(target)

        if (self.gate_patch[target] != 'I'):
            self.evaluate_patch()
        # 存储传递的角度值
        self.arguments[target] = phi / 2
        self.gate_patch[target] = 'RX'

    def RY(self, phi: float, target: int):
        """Add RY gate to the stack of current moment.
          Args:
            phi: 以弧度为单位的角度，对应于量子位状态围绕 Y 轴的给定 phi 值的旋转
        """

        self.check_input_one_gate(target)

        if (self.gate_patch[target] != 'I'):
            self.evaluate_patch()
        # 存储传递的角度值
        self.arguments[target] = phi / 2
        self.gate_patch[target] = 'RY'

    def RZ(self, phi: float, target: int):
        """Add RZ gate to the stack of current moment.
          Args:
            phi: 以弧度为单位的角度，对应于量子位状态围绕 z 轴的给定 phi 值的旋转
        """

        self.check_input_one_gate(target)

        if (self.gate_patch[target] != 'I'):
            self.evaluate_patch()
        # 存储传递的角度值
        self.arguments[target] = phi / 2
        self.gate_patch[target] = 'RZ'

    # = = = = = = = = = == = = = = = = = = = == = = = = = = = = = = = = = = = = = = == = =

    # 设置异常：检查许多量子比特门的不正确参数
    def check_input_control_gate(self, control: List, target: int):
        """"
          检查控制门的基本输入
          Args:
            control: a list of the controlled qubits
            target: 一个目标量子位
          Return: None.
          Raise:
            Value Errors.
        """
        if not isinstance(control, list):
            raise ValueError("Control must be a list.")
        if not len(control):
            raise ValueError("No control qubits has been provided.")
        if target > self.num_qubits - 1:
            raise ValueError("Qubit's index exceed the specidied size of the cirquit.")
        if target < 0 or not isinstance(target, int):
            raise ValueError("Target gate should be not-negative integer.")
        for control_qubit in control:
            if control_qubit > self.num_qubits - 1:
                raise ValueError("Qubit's index exceed the specidied size of the cirquit.")
            if control_qubit < 0 or not isinstance(control_qubit, int):
                raise ValueError("Control gate should be not-negative integer.")
        if target in control:
            raise ValueError("Target qubit was sent as a control.")
        if (not len(set(control)) == len(control)):
            raise ValueError("Control list contains repeating elements.")

    # 设置two qubits gates

    def CX(self, control: List, target: int):
        """Add CX (CNOT) gate to the stack of current moment.
          Args:
            control: 用作控制元素的量子位索引
            target: CX 门作用的量子比特节点的索引
          Returns: None.
          Raise: 如果 quibit 的索引超出电路大小，则出现 ValueError
                 如果目标和控制索引相等，则出现 ValueError
                 如果目标或控制 量子位 是浮点数或负数，则出现 ValueError
        """

        self.check_input_control_gate(control, target)
        # if gates applied on all quibits, evaluate current moment and start
        # to fill out next moment
        if (self.gate_patch[target] != 'I' or any(self.gate_patch[control_qubit] != 'I' for control_qubit in control)):
            self.evaluate_patch()
        self.gate_patch[target] = 'CX_Target_' + str(target)
        for control_qubit in control:
            self.gate_patch[control_qubit] = 'CX_Control_' + str(target)
        self.check_input_control_gate(control, target)
        self.control_gates_patch.append((control, target))

    def CZ(self, control: List, target: int):
        """Add CZ gate to the stack of current moment.
          Args:
            control: 用作控制元素的量子位索引
            target: CZ 门作用的量子比特节点的索引
        """

        self.check_input_control_gate(control, target)
        if (self.gate_patch[target] != 'I' or any(self.gate_patch[control_qubit] != 'I' for control_qubit in control)):
            self.evaluate_patch()
        self.gate_patch[target] = 'CZ_Target_' + str(target)
        for control_qubit in control:
            self.gate_patch[control_qubit] = 'CZ_Control_' + str(target)
        self.control_gates_patch.append((control, target))

    def CY(self, control: List, target: int):
        """Add CY gate to the stack of current moment.
          Args:
            control: 用作控制元素的量子位索引
            target: CY 门作用的量子比特节点的索引
        """

        self.check_input_control_gate(control, target)
        if (self.gate_patch[target] != 'I' or any(self.gate_patch[control_qubit] != 'I' for control_qubit in control)):
            self.evaluate_patch()
        self.gate_patch[target] = 'CY_Target_' + str(target)
        for control_qubit in control:
            self.gate_patch[control_qubit] = 'CY_Control_' + str(target)
        self.control_gates_patch.append((control, target))

    def CH(self, control: List, target: int):
        """Add CH gate to the stack of current moment.
          Args:
            control: 用作控制元素的量子位索引
            target: CH 门作用的量子比特节点的索引
        """

        self.check_input_control_gate(control, target)
        if (self.gate_patch[target] != 'I' or any(self.gate_patch[control_qubit] != 'I' for control_qubit in control)):
            self.evaluate_patch()

        self.gate_patch[target] = 'CH_Target_' + str(target)
        for control_qubit in control:
            self.gate_patch[control_qubit] = 'CH_Control_' + str(target)
        self.control_gates_patch.append((control, target))

    def CR(self, phi: float, control: List, target: int):
        """Add CR gate to the stack of current moment.
          Args:
            phi: 以弧度为单位的角度，对应于量子位状态围绕 z 轴的给定 phi 值的旋转
            control: 用作控制元素的量子位索引
            target: CR 门作用的量子比特节点的索引
        """

        self.check_input_control_gate(control, target)

        if (self.gate_patch[target] != 'I' or self.gate_patch[control] != 'I'):
            self.evaluate_patch()

        self.arguments[target] = np.exp(1.j * phi, dtype=np.complex128)
        self.gate_patch[target] = 'CR_Target_' + str(target)
        for control_qubit in control:
            self.gate_patch[control_qubit] = 'CR_Control_' + str(target)
        self.control_gates_patch.append((control, target))

    # 创建量子甲骨文(Oracle)，需要 Deutch 算法
    def Uf(self, func: callable):
        """
          创建一个与作为参数传递的函数等效的酉矩阵
          Args:
            func: 必须转换为酉矩阵的二元函数(function)
          Raise: 如果参数不可调用，则出现 ValueError
        """
        # callable, 可以判断传入的参数是否是可被调用的，如果返回True就说明传入的参数是一个函数
        if not callable(func):
            raise ValueError("Argument must be a function.")

        self.evaluate_patch()

        size = 2 ** self.num_qubits
        U = np.zeros((size, size), dtype=np.complex128)

        # 将二进制状态向量转换为整数
        def bin2int(bits):
            integer = 0
            for shift, j in enumerate(bits[::-1]):
                if j:
                    integer += 1 << shift
            return integer

        # 遍历每个状态并构建酉矩阵
        for state in product({0, 1}, repeat=self.num_qubits):
            x = state[:~0]
            y = state[~0]
            unitary_value = y ^ func(*x)  # 按位逻辑异或
            i = bin2int(state)
            j = bin2int(list(x) + [unitary_value])
            U[i, j] = 1.0 + 0.j

        # 加入张量网络
        self.network.append(tn.Node(U, backend=self.backend))
