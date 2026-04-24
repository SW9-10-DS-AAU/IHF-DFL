import numpy as np
import copy
from web3 import Web3
from openfl.ml.visualization import get_color

RNG = np.random.default_rng()


class Participant:
    def __init__(self, _id, _train, _val, _model, _optimizer, _criterion,
                 _attitude, _default_collateral, _max_collateral,
                 _attitudeSwitch=1, number_of_participants=None):
        self.id = _id
        self.train = _train
        self.val = _val
        self.model = _model
        self.previousModel = copy.deepcopy(_model)
        self.modelHash = Web3.solidity_keccak(['string'], [str(_model)]).hex()
        self.optimizer = _optimizer
        self.criterion = _criterion
        self.userToEvaluate = []
        self.currentAcc = 0
        # User's locally-trained model accuracy on their own validation set (after they trained on top of the global model).
        # Is set in: apply_training_results().
        self.currentLoss = 0
        # New variable introduced. Needs to be implemented in code. Alongside currentAcc.
        self.attitude = "good"
        self.futureAttitude = _attitude
        self.attitudeSwitch = _attitudeSwitch
        self.hashedModel = None
        self.address = None
        self.privateKey = None
        self.isRegistered = False
        # Old:  self.collateral = _default_collateral + np.random.randint(0,int(_max_collateral-_default_collateral))
        # ---- collateral (handles huge ranges; avoids int32 cap) ----
        lo = int(_default_collateral)
        hi = int(_max_collateral)
        if hi < lo:
            raise ValueError(f"max_collateral ({hi}) must be >= default_collateral ({lo})")

        diff = hi - lo
        jitter = int(RNG.integers(0, np.int64(diff), dtype=np.int64)) if diff > 0 else 0
        self.collateral = lo + jitter

        # ---- secret (big nonce) ----
        self.secret = int(RNG.integers(0, np.int64(10 ** 18), dtype=np.int64))
        # self.secret = np.random.randint(0,int(1e18))

        self.color = get_color(number_of_participants, self.attitude)
        self.roundRep = 0

        self.disqualified = False

        # INTERFACE VARIABLES - Not used for training. Only for reporting.
        self._accuracy = []  # User's accuracy on the global model. The actual accuracy evaluated on test set - is set in: finalize_user_evaluation().
        self._loss = []  # User's loss on the global model. The actual loss evaluated on test set - is set in: finalize_user_evaluation().
        self._globalrep = [self.collateral]
        self._roundrep = []
        self.last_attack_type = None  # Actual attack used this round (may differ from config due to fallbacks)

    def getStatus(self):
        user = f"$user${self.id}, {self.currentAcc}, {self.attitude}, {self.futureAttitude}, {self.attitudeSwitch}, {self.address}"
        return user
