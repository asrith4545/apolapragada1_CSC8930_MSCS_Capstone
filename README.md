PP²: Privacy-Preserving Intrusion Detection Framework

MS in Computer Science Capstone | Georgia State University

Project Overview

PP² is a privacy-preserving intrusion detection framework that combines Federated Learning (FL) and Differential Privacy (DP) to detect malicious network activity without exposing sensitive data.

Traditional intrusion detection systems rely on centralized data collection, which creates significant privacy risks. PP² addresses this challenge by keeping data localized at each client while still enabling collaborative model training. The system is evaluated using the UNSW-NB15 dataset across 10 distributed clients under both IID and non-IID data distributions.

Technical Approach

The framework uses Federated Learning with the FedAvg algorithm, where each client trains a local model and shares only model updates instead of raw network traffic. These updates are aggregated at a central server to build a global model.

To enhance privacy, Differential Privacy is applied through DP-SGD. This involves gradient clipping to limit the influence of individual data points and Gaussian noise injection to prevent information leakage from model updates.

The model itself is a feedforward Multilayer Perceptron with three hidden layers of sizes 256, 128, and 64. The task is formulated as a binary classification problem, distinguishing between normal and attack traffic. This formulation helps maintain stable convergence and handles class imbalance effectively in distributed environments.

Experimental Setup

The system simulates 10 distributed clients, each representing an independent organization. Training is performed over 70 communication rounds, with one local epoch per round. The model is optimized using Adam with a learning rate of 0.001 and a batch size of 512.

For Differential Privacy, a gradient clipping norm of 1.0 is used along with a noise multiplier of 0.6.

Results and Observations

The results demonstrate a clear privacy-utility trade-off. While introducing Differential Privacy leads to a modest drop in accuracy, the system maintains very strong detection capability.

One of the most important outcomes is the consistently high recall, around 0.98 across all configurations. This indicates that the model is highly effective at identifying attacks, minimizing false negatives, which is critical in cybersecurity applications.

Even in non-IID settings, where data distributions differ significantly across clients, the system remains robust and continues to perform well. This highlights the practicality of the approach in real-world distributed environments.

Conclusion

PP² shows that it is possible to build an effective intrusion detection system without centralizing sensitive data. By combining Federated Learning and Differential Privacy, the framework achieves strong attack detection performance while preserving privacy.

This makes it particularly suitable for environments where data sharing is restricted, such as healthcare, finance, and government systems.

Future Work

Future improvements can include incorporating a formal privacy budget using epsilon, implementing secure aggregation techniques, extending the model to multi-class classification, and evaluating performance on additional real-world datasets.

Author

Asrith Venkata Polapragada
MS in Computer Science
Georgia State University
