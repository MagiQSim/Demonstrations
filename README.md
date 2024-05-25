# About MagiQSim

At MagiQSim we specialize in classical simulation algorithms for universal quantum computation. In the field of theoretical quantum computing, understanding the reasons behind quantum advantage remains an open problem. This challenge can be rigorously investigated by developing classical simulation algorithms and analyzing their efficiency.

Stabilizer theory is key to unlocking many quantum features. In this formalism there are two primary classes of simulation algorithms: (1) stabilizer tableau simulation, such as Gottesmann-Knill simulation, which efficiently simulates stabilizer circuits, and (2) sampling algorithms, such as discrete Wigner functions, mainly applicable to qubits of odd local dimensions. Recently, a new sampling algorithm, known as the $\Lambda$ simulation, was developed using operator-theoretic polytopes. The distinguishing feature of this simulation algorithm is that quantum states and operations are represented positively, i.e., by a probability distribution. In contrast, previous sampling algorithms generated quasi-probability distributions that could assume negative values. The presence of negativity in quasi-probability representations is identified as a source of quantum speedup. However, in the $\Lambda$ simulation, this negativity is absent, and the source of quantum speedup is currently under theoretical investigation.

Our core technology leverages classical simulation algorithms, such as $\Lambda$ simulation (MagiQ Simulator), to tackle computationally hard problems. Our methods utilize cutting-edge theoretical techniques from polytope theory and algebraic topology. By employing these advanced and unique tools, we enhance existing simulation algorithms and explore new ones.

# MagiQ Simulator

The main piece of this repository is the jupyter notebook MagiQSim_demos.ipynb which demonstrates a prototype of our MagiQ Simulator based on closed-noncontextual, or cnc MagiQ Keys. In particular we give a few toy examples of what is possible with our MagiQ Simulator which can be used to classically simulate quantum circuits. Eventually our MagiQ Simulator will be used for benchmarking current NISQ circuits and provide a platform for developing and testing quantum algorithms.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Features

- High-performance classical simulation of quantum circuits
- Integration with existing quantum computing frameworks like Qiskit

## Contributing

We welcome contributions! For more details on how to contribute to this project please use the contact below.

## Contact

For any questions or inquiries, please contact merlinmagiqsquare [at] gmail [dot] com.