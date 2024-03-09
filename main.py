from pathlib import Path
import pickle
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config_fn, get_evaluate_fn
import flwr as fl


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare your dataset
    trainloaders, validationloaders, testloaders = prepare_dataset(cfg.num_clients,
                                                                   cfg.batch_size)

    ## 3. Define your clients
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ## 4. Define your strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.00001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.00001,
        min_evaluate_clients=cfg.num_clients_per_round_evaluate,
        min_available_clients=cfg.num_clients,
        on_fit_config_fn=get_on_fit_config_fn(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloaders))

    ## 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        #client_resources={"num_cpus": 1, "num_gpus": 0.25}
    )

    ## 6. Save your results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / 'results.pkl'

    results = {
        'history': history,
        'anythingelse': 'here'
    }

    with open(str(results_path), 'wb') as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()