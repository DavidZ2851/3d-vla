from .denoise_actor_3d import DenoiseActor as DenoiseActor3D
from .denoise_actor_2d import DenoiseActor as DenoiseActor2D
from .denoise_actor_3d_2dwrist import DenoiseActor as DenoiseActor2Dwrist
from .flower.model import FLOWERVLA
from .flower.model_peract2 import FLOWERVLA as FLOWERVLARLB
from .flower.model_peract2_3d import FLOWERVLA3D
from .flower_actor import FlowerActor3D, FlowerActor3DGroundingCond, FlowerActor2D
from .flower_actor_qwen import QwenFlowerActor3D
from .flower_actor_qwen_stagewise import QwenFlowerActor3DStagewise
from .flower_actor_qwen_grounding import (
    QwenFlowerActor3DGroundingCond,
    QwenFlowerActor3DGroundingCondLLMVisual,
    QwenFlowerActor3DGroundingCondQwen3D,
    QwenFlowerActor3DGroundingPred,
)
try:
    from .qwen_pi_llm_actor_3d import QwenPi0LLMActor3D, QwenPi05LLMActor3D
    _QWEN_PI_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    QwenPi0LLMActor3D = None
    QwenPi05LLMActor3D = None
    _QWEN_PI_IMPORT_ERROR = exc
from .qwen_dualexpert_3dfa_actor import QwenDualExpert3DFAActor
# from .old_diffuser_actor import DiffuserActor


def fetch_model_class(model_type):
    if model_type == 'denoise3d':  # standard 3DFA
        return DenoiseActor3D
    if model_type == 'denoise2d':  # standard 2DFA
        return DenoiseActor2D
    if model_type == 'denoise3d_2dwrist':  # 3D front + 2D wrist
        return DenoiseActor2Dwrist
    if model_type == 'flower':
        return FLOWERVLA
    if model_type == 'flower_rlbench':
        return FLOWERVLARLB
    if model_type == 'flower_rlbench_3d':
        return FLOWERVLA3D
    if model_type == 'flower_actor_3d':
        return FlowerActor3D
    if model_type == 'flower_actor_3d_grounding_cond':
        return FlowerActor3DGroundingCond
    if model_type == 'flower_actor_2d':
        return FlowerActor2D
    if model_type == 'qwen_flower_actor_3d':
        return QwenFlowerActor3D
    if model_type == 'qwen_flower_actor_3d_stagewise':
        return QwenFlowerActor3DStagewise
    if model_type == 'qwen_flower_actor_3d_grounding_cond':
        return QwenFlowerActor3DGroundingCond
    if model_type == 'qwen_flower_actor_3d_grounding_cond_llm_visual':
        return QwenFlowerActor3DGroundingCondLLMVisual
    if model_type == 'qwen_flower_actor_3d_grounding_cond_qwen3d':
        return QwenFlowerActor3DGroundingCondQwen3D
    if model_type == 'qwen_flower_actor_3d_grounding_pred':
        return QwenFlowerActor3DGroundingPred
    if model_type == 'qwen_pi0_llm_actor_3d':
        if QwenPi0LLMActor3D is None:
            raise ImportError(
                "qwen_pi0_llm_actor_3d requires optional Qwen3-VL dependencies "
                "(e.g. `transformers.models.qwen3_vl`) that are unavailable in "
                "this environment."
            ) from _QWEN_PI_IMPORT_ERROR
        return QwenPi0LLMActor3D
    if model_type == 'qwen_pi05_llm_actor_3d':
        if QwenPi05LLMActor3D is None:
            raise ImportError(
                "qwen_pi05_llm_actor_3d requires optional Qwen3-VL dependencies "
                "(e.g. `transformers.models.qwen3_vl`) that are unavailable in "
                "this environment."
            ) from _QWEN_PI_IMPORT_ERROR
        return QwenPi05LLMActor3D
    if model_type == 'qwen_dualexpert_3dfa_actor':
        return QwenDualExpert3DFAActor
    # if model_type == '3dda':
    #     return DiffuserActor
    return None
