from utils.run_utils import run_command
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


log_fp = "results/test_global.log"
device = "cuda:0"
datasets = ['acm']
atk_names = ['HetePRBCD']
atk_rates = [0]
hete_models = [ 'PSTABLE']

for dataset in datasets:
    for atk_name in atk_names:
        for atk_rate in atk_rates:
            # 构造公共参数
            common_cmd_args = [
                "--dataname", dataset,
                "--atk_name", atk_name,
                "--atk_rate", str(atk_rate),
                "--log_fp", log_fp,
                "--device", device
            ]

            # # Our method
            # run_command([
            #     "python3", "-u", "our_global.py",
            # ] + common_cmd_args)

            #Homo models
            # for homo_model in homo_models:
            #     run_command([
            #         "python3", "-u", "homo_global.py",
            #         "--model", homo_model,
            #     ] + common_cmd_args)

            # Hete models
            for hete_model in hete_models:
                cmd = [
                    "python3", "-u", "-m", "core.hete_global",
                    "--model", hete_model,
                ]

                run_command(cmd + common_cmd_args)