import hydra
from omegaconf import DictConfig

from learning.inference_3d import Inference3D
from manipulation.experiment_real import ExperimentReal
from manipulation.experiment_virtual import ExperimentVirtual


@hydra.main(
    config_path="config/real_experiment", config_name="experiment_real_tshirt_long", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    # create experiment instance that contains robot controller, camera, environment setup, etc.
    # exp = ExperimentReal(config=cfg.experiment)
    exp = ExperimentVirtual(config=cfg.experiment)  # uncomment this line to use virtual environment

    # the interface of AI model (UFONet)
    inference = Inference3D(experiment=exp, **cfg.inference)

    # capture point cloud
    obs, err = exp.capture_pcd()  # please re-implement this method if you are using custom hardware

    # predict action type
    action_type = inference.predict_raw_action_type(obs)

    # predict action
    prediction_message, action_message, err = inference.predict_action(obs, action_type)

    # execute action
    err = exp.execute_action(action_message)  # please re-implement this method if you are using custom hardware


if __name__ == "__main__":
    main()