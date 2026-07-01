This project provides the implementation of physics-data fusion methods for online State-of-Energy estimation of electric vehicle batteries under complex real-world operating conditions. The main proposed method is an Error Compensation Mechanism that combines the Adaptive Moving-Window Extended Kalman Filter with PatchFormer, where AMWEKF provides physically interpretable initial SOE estimates and PatchFormer learns residual errors for data-driven compensation. The repository also includes the implementation of a strongly coupled Physics-Informed Neural Network model for comparison.

We have released:

- Data preprocessing code.
- The core PINN framework.
- The error compensation mechanism.
