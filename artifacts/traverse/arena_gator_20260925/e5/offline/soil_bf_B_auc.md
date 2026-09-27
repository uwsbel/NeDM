## auto (holdout-mode models: dev fold + val; deploy-mode models: val)
| model | mode | rows used | f104_gator_all: start AUC / moving AUC / pick fail (groups) | f104_hmmwv_all: start AUC / moving AUC / pick fail (groups) | f104_gator_t06: start AUC / moving AUC / pick fail (groups) | f104_hmmwv_t06: start AUC / moving AUC / pick fail (groups) |
|---|---|---|---|---|---|---|
| G_full_deploy | deploy | val | 0.950 / 0.962 / 0.643 (56) | 0.822 / 0.838 / 0.357 (56) | 0.941 / 0.949 / 0.661 (56) | 0.814 / 0.819 / 0.446 (56) |
| G_full_holdout | holdout | dev+val | 0.946 / 0.959 / 0.632 (280) | 0.814 / 0.847 / 0.350 (280) | 0.944 / 0.951 / 0.679 (280) | 0.826 / 0.854 / 0.429 (280) |
| H_full_deploy | deploy | val | 0.847 / 0.876 / 0.714 (56) | 0.991 / 0.985 / 0.179 (56) | 0.782 / 0.829 / 0.786 (56) | 0.995 / 0.984 / 0.232 (56) |
| H_full_holdout | holdout | dev+val | 0.827 / 0.850 / 0.743 (280) | 0.979 / 0.979 / 0.204 (280) | 0.828 / 0.845 / 0.736 (280) | 0.983 / 0.980 / 0.254 (280) |
| G_deploy | deploy | val | 0.952 / 0.949 / 0.679 (56) | 0.832 / 0.841 / 0.250 (56) | 0.957 / 0.935 / 0.661 (56) | 0.829 / 0.833 / 0.375 (56) |
| G_holdout | holdout | dev+val | 0.936 / 0.951 / 0.643 (280) | 0.811 / 0.846 / 0.336 (280) | 0.925 / 0.936 / 0.686 (280) | 0.822 / 0.855 / 0.429 (280) |
| H_deploy | deploy | val | 0.849 / 0.868 / 0.750 (56) | 0.971 / 0.974 / 0.196 (56) | 0.798 / 0.819 / 0.786 (56) | 0.977 / 0.971 / 0.268 (56) |
| H_holdout | holdout | dev+val | 0.823 / 0.848 / 0.721 (280) | 0.974 / 0.973 / 0.200 (280) | 0.812 / 0.833 / 0.746 (280) | 0.975 / 0.971 / 0.271 (280) |

## val groups only (every model)
| model | mode | rows used | f104_gator_all: start AUC / moving AUC / pick fail (groups) | f104_hmmwv_all: start AUC / moving AUC / pick fail (groups) | f104_gator_t06: start AUC / moving AUC / pick fail (groups) | f104_hmmwv_t06: start AUC / moving AUC / pick fail (groups) |
|---|---|---|---|---|---|---|
| G_full_deploy | deploy | val | 0.950 / 0.962 / 0.643 (56) | 0.822 / 0.838 / 0.357 (56) | 0.941 / 0.949 / 0.661 (56) | 0.814 / 0.819 / 0.446 (56) |
| G_full_holdout | holdout | val | 0.950 / 0.955 / 0.643 (56) | 0.825 / 0.843 / 0.304 (56) | 0.936 / 0.951 / 0.679 (56) | 0.819 / 0.842 / 0.464 (56) |
| H_full_deploy | deploy | val | 0.847 / 0.876 / 0.714 (56) | 0.991 / 0.985 / 0.179 (56) | 0.782 / 0.829 / 0.786 (56) | 0.995 / 0.984 / 0.232 (56) |
| H_full_holdout | holdout | val | 0.847 / 0.867 / 0.732 (56) | 0.983 / 0.982 / 0.214 (56) | 0.782 / 0.820 / 0.768 (56) | 0.982 / 0.980 / 0.250 (56) |
| G_deploy | deploy | val | 0.952 / 0.949 / 0.679 (56) | 0.832 / 0.841 / 0.250 (56) | 0.957 / 0.935 / 0.661 (56) | 0.829 / 0.833 / 0.375 (56) |
| G_holdout | holdout | val | 0.949 / 0.955 / 0.679 (56) | 0.829 / 0.846 / 0.232 (56) | 0.952 / 0.944 / 0.679 (56) | 0.834 / 0.850 / 0.393 (56) |
| H_deploy | deploy | val | 0.849 / 0.868 / 0.750 (56) | 0.971 / 0.974 / 0.196 (56) | 0.798 / 0.819 / 0.786 (56) | 0.977 / 0.971 / 0.268 (56) |
| H_holdout | holdout | val | 0.857 / 0.868 / 0.696 (56) | 0.969 / 0.975 / 0.196 (56) | 0.809 / 0.822 / 0.732 (56) | 0.972 / 0.974 / 0.250 (56) |
