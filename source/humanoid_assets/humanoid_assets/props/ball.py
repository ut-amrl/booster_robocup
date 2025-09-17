from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObjectCfg

# Size-5 soccer ball: diameter ~= 0.22 m -> radius ~= 0.11 m
BALL_CFG = DeformableObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.MeshSphereCfg(
        radius=0.11,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(
            deformable_enabled=True,
            self_collision=True,
            # earlier contact helps grabbing / bouncing
            contact_offset=0.008,  # try 0.005-0.01
            rest_offset=0.0,
            # stability / feel
            solver_position_iteration_count=80,
            vertex_velocity_damping=0.002,
            # FEM resolution (perf vs. fidelity)
            simulation_hexahedral_resolution=48,
        ),
        physics_material=sim_utils.DeformableBodyMaterialCfg(
            # mass ~= density * volume; for r=0.11 -> ~0.00558 m^3
            # 0.43 kg / 0.00558 ~= 77 kg/m^3 (hollow ball approximated as a soft volume)
            density=77.0,
            # stiffness & volume preservation ~= "inflation"
            youngs_modulus=2.0e7,  # try 1e7-5e7
            poissons_ratio=0.48,  # near-incompressible
            # energy loss / squishiness
            elasticity_damping=0.002,
            damping_scale=0.6,
            # surface slide vs. grip
            dynamic_friction=0.35,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
    ),
    init_state=DeformableObjectCfg.InitialStateCfg(pos=(14.0, -16.0, 15.0)),
    debug_vis=False,
)
