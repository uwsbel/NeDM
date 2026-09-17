// Skeptic check for claim C12: time ChVulkanRTScene::SyncFromSystem (the real library function from the local
// Chrono build) on a stand-in scene: one fixed 512x512 heightmap mesh (522,242 triangles, per-vertex normals + UVs,
// optional texture path in the material, like RigidTerrain::Patch::SetTexture) plus 20 other bodies.
// Modes for the 5 "vehicle" bodies that move between calls:
//   empty : they carry a visual model that was emptied (what SetXVisualizationType(NONE) leaves behind)
//   null  : they never had a visual model (bodies effectively out of the rendered system)
//   mesh  : they carry a small visible mesh (full vehicle visible -> revision bump each move)
// Also times the no-motion call (signature walk + early return).
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include "chrono/physics/ChSystemNSC.h"
#include "chrono/physics/ChBody.h"
#include "chrono/assets/ChVisualShapeTriangleMesh.h"
#include "chrono/assets/ChVisualShapeBox.h"
#include "chrono/geometry/ChTriangleMeshConnected.h"
#include "chrono_sensor/vulkan/ChVulkanRTScene.h"

using namespace chrono;
using namespace chrono::sensor;
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "empty";
    bool tex = argc > 2 && std::string(argv[2]) == "tex";
    ChSystemNSC sys;
    auto mesh = chrono_types::make_shared<ChTriangleMeshConnected>();
    const int N = 512;
    auto& V = mesh->GetCoordsVertices();
    auto& NN = mesh->GetCoordsNormals();
    auto& UV = mesh->GetCoordsUV();
    auto& F = mesh->GetIndicesVertices();
    auto& FN = mesh->GetIndicesNormals();
    auto& FU = mesh->GetIndicesUV();
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++) {
            V.push_back(ChVector3d(i * 80.0 / (N - 1) - 40, j * 80.0 / (N - 1) - 40, 0.5 * std::sin(i * 0.1) * std::cos(j * 0.07)));
            NN.push_back(ChVector3d(0, 0, 1));
            UV.push_back(ChVector2d(i / double(N - 1), j / double(N - 1)));
        }
    for (int j = 0; j < N - 1; j++)
        for (int i = 0; i < N - 1; i++) {
            int v0 = j * N + i;
            ChVector3i a(v0, v0 + N + 1, v0 + N), b(v0, v0 + 1, v0 + N + 1);
            F.push_back(a); FN.push_back(a); FU.push_back(a);
            F.push_back(b); FN.push_back(b); FU.push_back(b);
        }
    auto terrain = chrono_types::make_shared<ChBody>();
    terrain->SetFixed(true);
    auto shape = chrono_types::make_shared<ChVisualShapeTriangleMesh>();
    shape->SetMesh(mesh);
    auto mat = chrono_types::make_shared<ChVisualMaterial>();
    if (tex) {
        mat->SetKdTexture("/home/harry/NeDM-traverse_mppi/chrono/data/sensor/textures/grass_texture.jpg");
        mat->SetTextureScale(40, 40);
    }
    shape->AddMaterial(mat);
    terrain->AddVisualShape(shape);
    sys.AddBody(terrain);

    std::vector<std::shared_ptr<ChBody>> movers;
    for (int k = 0; k < 20; k++) {
        auto b = chrono_types::make_shared<ChBody>();
        sys.AddBody(b);
        if (k < 5) {
            movers.push_back(b);
            if (mode == "empty") {
                b->AddVisualShape(chrono_types::make_shared<ChVisualShapeBox>(1, 1, 1));
                b->GetVisualModel()->Clear();
            } else if (mode == "mesh") {
                b->AddVisualShape(chrono_types::make_shared<ChVisualShapeBox>(1, 1, 1));
            }
        }
    }
    std::cout << "mode=" << mode << " tex=" << tex << " faces=" << F.size() << " sizeof(tri)=" << sizeof(ChVulkanRTTriangle) << "\n";
    ChVulkanRTScene scene;
    auto t0 = clk::now();
    scene.SyncFromSystem(&sys);
    auto t1 = clk::now();
    std::cout << "initial sync " << std::chrono::duration<double>(t1 - t0).count() << " s rev " << scene.GetRevision()
              << " visible_shapes " << scene.GetStats().visible_shapes << "\n";
    for (int it = 0; it < 3; it++) {
        auto a = clk::now();
        scene.SyncFromSystem(&sys);
        auto b = clk::now();
        std::cout << "no-motion sync " << std::chrono::duration<double>(b - a).count() * 1e6 << " us rev " << scene.GetRevision() << "\n";
    }
    for (int it = 0; it < 4; it++) {
        for (auto& m : movers) m->SetPos(ChVector3d(0.01 * (it + 1), 0, 0));
        auto a = clk::now();
        scene.SyncFromSystem(&sys);
        auto b = clk::now();
        std::cout << "moved sync " << std::chrono::duration<double>(b - a).count() << " s rev " << scene.GetRevision() << "\n";
    }
    return 0;
}
