import pennylane as qml



def CZRXRYLayer(params, n_qubits):
    for i in range(n_qubits -1):
        qml.CZ(wires = [i, i+1])
    qml.CZ(wires = [n_qubits - 1, 0])

    for i in range(n_qubits):
        qml.RX(params[i, 0], wires = i)
        qml.RY(params[i, 1], wires = i)