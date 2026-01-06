import omni
from pxr import UsdGeom, UsdPhysics
from omni.isaac.core import World
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

stage = omni.usd.get_context().get_stage()

# ---- Create a static table (collider only) ----
table_path = "/World/table"
create_prim(table_path, "Cube", position=[0.6, 0.0, 0.4], scale=[1.2, 0.8, 0.05])

table_prim = stage.GetPrimAtPath(table_path)
cube_geom = UsdGeom.Cube(table_prim)
cube_geom.GetSizeAttr().Set(1.0)  # make scaling predictable (meters)

UsdPhysics.CollisionAPI.Apply(table_prim)  # static collider (no rigid body)

# ---- Create a dynamic cube above the table ----
# Table top z = 0.4 + (0.05/2) = 0.425
# Cube center z = top + (0.05/2) = 0.45
world.scene.add(
    DynamicCuboid(
        prim_path="/World/Target_object",
        name="Target_object",
        position=[0.6, 0.0, 0.45],
        size=0.05,
        mass=0.2
    )
)

world.reset()
print("Created /World/table (static collider) and /World/Target_object (dynamic). Press Play.")
