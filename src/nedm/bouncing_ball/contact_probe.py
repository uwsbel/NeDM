"""Read-only physics diagnosis of contact-normal jitter in the approved scene.

These traces do not train or select a model and do not change any contact setting.
"""
import argparse
import math
from pathlib import Path
from nedm.bouncing_ball.collection import atomic_json,build_scene,load_config,provenance


def main():
    import pychrono as chrono
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    config=load_config(Path('configs/bouncing_ball/chrono_v1.json'))
    dt=config['simulation']['step_s']
    gravity=config['simulation']['gravity_mps2']
    class Reporter(chrono.ReportContactCallback):
        def __init__(self):
            super().__init__()
            self.contacts=[]
        def OnReportContact(self,pa,pb,plane,distance,radius,force,torque,a,b,offset):
            if abs(force.x)>1:
                n=plane.GetAxisX()
                self.contacts.append(dict(normal=[n.x,n.y,n.z],distance=distance,force=[force.x,force.y,force.z],
                                          pA=[pa.x,pa.y,pa.z],pB=[pb.x,pb.y,pb.z]))
            return True
    entries=[]
    for vx in (4.2,5.5,6.7):
        for vz in (-10.3,-9.7,-9.1):
            for change in (0.,1e-6):
                system,ball,floor,wall=build_scene(config,vx+change,vz)
                events=[]
                for step in range(1,math.ceil(1.7/dt)+1):
                    pre=ball.GetPosDt()
                    pre_values=[pre.x,pre.z]
                    system.DoStepDynamics(dt)
                    ground=abs(floor.GetContactForce().z)>1
                    side=abs(wall.GetContactForce().x)>1
                    if not (ground or side):continue
                    reporter=Reporter()
                    system.GetContactContainer().ReportAllContacts(reporter)
                    post=ball.GetPosDt()
                    event={'kind':'ground' if ground else 'wall','time_s':step*dt,'pre_velocity_mps':pre_values,
                           'post_velocity_mps':[post.x,post.z],'contacts':reporter.contacts}
                    if ground:
                        event['axis_normal_impulse_discrepancy_mps']=(post.z-pre_values[1]+gravity*dt)-(-(1+config['contact']['restitution'])*pre_values[1]+gravity*dt)
                    events.append(event)
                    if side:break
                entries.append({'launch_velocity_mps':[vx+change,vz],'events':events})
    atomic_json(args.output,{'diagnostic_only':True,'scene_unchanged':True,'not_used_for_training_or_selection':True,
                            'runtime':provenance(),'episodes':entries})
    print('contact-normal probe complete:',len(entries),'launches',flush=True)


if __name__=='__main__':main()
