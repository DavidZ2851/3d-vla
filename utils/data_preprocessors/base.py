class DataPreprocessor:

    def __init__(self, keypose_only=False, num_history=1,
                 custom_imsize=None, depth2cloud=None):
        self.keypose_only = keypose_only
        self.num_history = num_history
        self.custom_imsize = custom_imsize
        self.depth2cloud = depth2cloud

    def process_actions(self, actions):
        """Action shape: (B, T, nhand, 3+rot+1)."""
        actions = actions.cuda(non_blocking=True)
        if self.keypose_only:
            actions = actions[:, [-1]]
        return actions

    def process_proprio(self, proprio):
        """Proprio shape: (B, nhist, nhand, 3+rot+1)."""
        proprio = proprio.cuda(non_blocking=True)
        nhist_ = proprio.size(1)
        assert nhist_ >= self.num_history, "not enough proprio timesteps"
        # the first proprio is the current state
        proprio = proprio[:, :max(self.num_history, 1)]
        return proprio

    def process_joint_action(self, joint_action):
        """Joint action shape: (B, T, nhand, n_joint+1). Used only by the
        joint-action model; a passthrough-to-cuda mirror of process_actions."""
        joint_action = joint_action.cuda(non_blocking=True).float()
        if self.keypose_only:
            joint_action = joint_action[:, [-1]]
        return joint_action

    def process_joint_proprio(self, joint_proprio):
        """Joint proprio shape: (B, nhist, nhand, n_joint+1). Mirror of
        process_proprio for the joint-action model."""
        joint_proprio = joint_proprio.cuda(non_blocking=True).float()
        nhist_ = joint_proprio.size(1)
        assert nhist_ >= self.num_history, "not enough joint proprio timesteps"
        joint_proprio = joint_proprio[:, :max(self.num_history, 1)]
        return joint_proprio

    def process_obs(self, rgbs, pcds):
        pass
