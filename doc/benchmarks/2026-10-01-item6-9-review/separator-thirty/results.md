# Separator campaign: all thirty runs

Inputs in answer order: hello world, hello there, loving world, loving there. Target: 0, 1, 1, 0.

| Gate | Run | Answers | MSE | Correct | At class bar | Read-backs | Reconstructed | Contrast |
|---|---:|---|---:|---:|---|---|---:|---:|
| sum | 1 | 0.4994552, 0.4999889, 0.5010812, 0.5016149 | 0.25000101 | 2 | False | there there; world world; there there; world world | 0/4 | 0 |
| sum | 2 | 0.5056681, 0.5050159, 0.5012563, 0.5006041 | 0.25001478 | 2 | False | world world; loving loving; world world; hello hello | 0/4 | 0 |
| sum | 3 | 0.4998222, 0.5013635, 0.5002961, 0.5018374 | 0.25000134 | 3 | False | there there; world world; there there; world world | 0/4 | 2.98023224e-08 |
| sum | 4 | 0.5017311, 0.5018351, 0.5018432, 0.5019472 | 0.2500034 | 2 | False | there there; world world; there there; world world | 0/4 | 0 |
| sum | 5 | 0.4981998, 0.4987136, 0.4982177, 0.4987315 | 0.25000244 | 2 | False | there there; world world; there there; world world | 0/4 | 0 |
| sum | 6 | 0.4991674, 0.4965186, 0.5053078, 0.502659 | 0.25001201 | 2 | False | hello hello; loving loving; loving loving; hello hello | 0/4 | -5.96046448e-08 |
| sum | 7 | 0.4979032, 0.4986456, 0.4985674, 0.4993097 | 0.25000221 | 2 | False | hello hello; hello hello; there there; world world | 0/4 | 0 |
| sum | 8 | 0.5005034, 0.4999885, 0.5004304, 0.4999155 | 0.25000012 | 2 | False | hello hello; loving loving; hello hello; there there | 0/4 | 2.98023224e-08 |
| sum | 9 | 0.5015289, 0.500726, 0.5011733, 0.5003703 | 0.25000107 | 2 | False | there there; world world; there there; world world | 0/4 | -5.96046448e-08 |
| sum | 10 | 0.499102, 0.4993148, 0.5018958, 0.5021086 | 0.25000232 | 2 | False | loving loving; hello hello; hello hello; there there | 0/4 | 0 |
| class | 1 | 0.2105753, 0.7130052, 0.7762489, 0.2644344 | 0.061674517 | 4 | False | hello hello; there hello; world hello; hello there | 1/4 | -1.01424441 |
| class | 2 | 0.003052384, 0.9994431, 0.9969947, -0.001003027 | 4.916214e-06 | 4 | True | world hello; hello hello; world world; there hello | 1/4 | -1.99438849 |
| class | 3 | 0.1663118, 0.8759402, 0.7961053, 0.09776157 | 0.023545209 | 4 | True | world world; there world; loving world; there loving | 2/4 | -1.4079721 |
| class | 4 | 0.3314974, 0.8462671, 0.8952962, 0.07194823 | 0.037415944 | 4 | True | world hello; there there; loving world; there loving | 3/4 | -1.33811766 |
| class | 5 | 0.1150433, 0.8249167, 0.8286349, 0.1677477 | 0.025348604 | 4 | True | world world; loving loving; world loving; loving loving | 1/4 | -1.37076059 |
| class | 6 | 0.1629775, 0.7229828, 1.013205, 0.1750522 | 0.033529464 | 4 | True | world there; there loving; world world; world there | 0/4 | -1.39815763 |
| class | 7 | 0.1473702, 0.8002793, 0.8360678, 0.2349477 | 0.035920136 | 4 | True | world world; there there; hello there; loving there | 1/4 | -1.25402912 |
| class | 8 | 0.1306226, 0.9603635, 0.7163662, 0.05807203 | 0.025613446 | 4 | True | loving world; there there; loving world; loving there | 2/4 | -1.48803514 |
| class | 9 | 0.3065828, 0.7162505, 0.7663263, 0.3244353 | 0.083592102 | 4 | False | there world; loving world; world there; there there | 0/4 | -0.851558745 |
| class | 10 | 0.1751112, 0.7478436, 0.9089317, 0.2994011 | 0.048045315 | 4 | True | world hello; hello there; world hello; there hello | 2/4 | -1.18226296 |
| reconstruction | 1 | 0.3104902, 0.7579664, 0.714164, 0.3053867 | 0.082486928 | 4 | False | world loving; world loving; hello world; hello world | 0/4 | -0.856253475 |
| reconstruction | 2 | 0.09098989, 0.9513792, 0.9137155, 0.1129362 | 0.007710685 | 4 | True | hello world; there hello; loving world; loving there | 4/4 | -1.66116863 |
| reconstruction | 3 | 0.09321517, 0.9216313, 0.9304362, 0.09224075 | 0.007044551 | 4 | True | hello hello; hello there; loving loving; there loving | 2/4 | -1.66661155 |
| reconstruction | 4 | 0.1752646, 0.751016, 0.7090138, 0.2077474 | 0.055135671 | 4 | False | hello world; hello there; world loving; hello loving | 3/4 | -1.07701772 |
| reconstruction | 5 | 0.03944075, 0.8401744, 0.8841501, 0.2969861 | 0.032180429 | 4 | True | world there; there there; world loving; loving there | 2/4 | -1.38789773 |
| reconstruction | 6 | -0.0001356006, 1.000831, 0.9964287, -0.005150437 | 9.9973386e-06 | 4 | True | world hello; there hello; world there; there there | 2/4 | -2.00254542 |
| reconstruction | 7 | 0.09591839, 0.8793654, 0.9010757, 0.09979746 | 0.010874645 | 4 | True | loving loving; loving there; hello hello; hello there | 0/4 | -1.58472532 |
| reconstruction | 8 | 0.1749813, 0.7304274, 0.9456954, 0.1564083 | 0.032675099 | 4 | True | hello world; there hello; hello world; hello world | 2/4 | -1.34473318 |
| reconstruction | 9 | 0.1321478, 0.8271164, 0.97605, 0.2802697 | 0.03161912 | 4 | True | world world; world there; world loving; loving there | 2/4 | -1.39074886 |
| reconstruction | 10 | 0.05320567, 0.9209642, 0.8752935, 0.0954504 | 0.008434996 | 4 | True | world there; there there; loving world; loving there | 2/4 | -1.64760166 |

Every unavailable flag is false. The JSON retains full precision, inputs, process guards, times and source hashes.
No new tolerance defines “one half”: the sum answers are reported numerically (maximum deviation 0.005668104; contrast at most 5.96e-8).
