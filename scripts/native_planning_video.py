"""Native Habitat RGB/map/video utilities with read-only EventWM overlays.

All policy telemetry comes from the original recorded rollout. This module
never constructs robot actions, calls a model, or returns policy observations.
"""
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from functools import lru_cache

BG=(18,24,33);FG=(233,238,246);MUTED=(154,168,186)
CYAN=(52,220,239);YELLOW=(255,208,67);GREEN=(87,220,142);RED=(251,105,113)
BLUE=(81,150,247);GRAY=(111,129,148);WHITE=(248,249,253);MAGENTA=(236,126,221)
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


@lru_cache(maxsize=24)
def font(size):return ImageFont.truetype(FONT,size)


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]


def text(image,value,xy,size=18,color=FG):
    canvas=Image.fromarray(image);draw=ImageDraw.Draw(canvas)
    draw.text(xy,str(value),font=font(size),fill=color)
    image[:]=np.asarray(canvas)


class EpisodeView:
    def __init__(self,args,policy,tag):
        import habitat_sim
        from habitat.utils.visualizations import maps
        from hashlib import sha256
        from omegaconf import OmegaConf
        self.args,self.policy,self.tag=args,policy,tag
        self.directory=args.output/'replays'/policy/tag
        self.validation=read(self.directory/'validation.json');assert self.validation['passed']
        source=args.source/'runs'/policy/tag
        self.metrics=read(source/'metrics.json');self.actions=lines(source/'actions.jsonl')
        self.decisions=lines(source/'decisions.jsonl');self.frames=read(self.directory/'frames.json')
        self.times=np.asarray([f['timestamp'] for f in self.frames])
        self.truth={round(v['timestamp'],8):v['humans'] for v in read(source/'perception_audit.json')}
        self.row=self.validation['source_provenance']['row'];self.goal=np.asarray(self.metrics['goal'])
        self.clock=read(source/'clock.json');self.maps=maps
        cfg=OmegaConf.load(self.directory/'render_config.yaml').habitat.simulator.agents.agent_0
        identity=sha256((self.row['scene_path']+repr((cfg.radius,cfg.height,cfg.max_climb,cfg.max_slope))).encode()).hexdigest()[:20]
        self.pf=habitat_sim.PathFinder();assert self.pf.load_nav_mesh(str(args.output/'navmeshes'/(identity+'.navmesh')))
        self.mpp=maps.calculate_meters_per_pixel(1024,pathfinder=self.pf)
        self.raw_map=maps.get_topdown_map(self.pf,self.metrics['start_position'][1],1024,True)
        self.native_map=maps.colorize_topdown_map(self.raw_map)
        # Native occupancy/border geometry and projection are retained. A dark
        # display palette improves overlay contrast; no geometry is synthesized.
        for code,color in ((0,(25,31,41)),(1,(64,72,84)),(2,(108,124,141))):
            self.native_map[self.raw_map==code]=color

    def frame_index(self,t):return max(0,min(len(self.frames)-1,int(np.searchsorted(self.times,t+1e-8,side='right')-1)))

    def plan(self,index):
        past=[d for d in self.decisions if d['candidates'] and d['step']<=index]
        return past[-1] if past else None

    def rgb(self,index,kind='third'):
        p=self.directory/'frames'/f'{index:04d}_{kind}.jpg'
        value=cv2.imread(str(p));assert value is not None,str(p)
        return cv2.cvtColor(value,cv2.COLOR_BGR2RGB)

    def status(self,index):
        if index==len(self.frames)-1:
            return 'GOAL SUCCESS' if self.metrics['navigation_success'] else 'HUMAN COLLISION',GREEN if self.metrics['navigation_success'] else RED
        return 'NAVIGATING',FG

    def local_map(self,index,width=512,height=354):
        frame=self.frames[index];p=np.asarray(frame['robot_position']);plan=self.plan(index)
        canvas=np.full((height,width,3),BG,np.uint8)
        title='Habitat local NavMesh + legal observations' if not self.args.show_ground_truth else 'Habitat NavMesh | legal tracks + DIAGNOSTIC GT'
        text(canvas,title,(14,8),18 if not self.args.show_ground_truth else 16)
        def grid(position):
            row,col=self.maps.to_grid(position[2],position[0],self.raw_map.shape,pathfinder=self.pf)
            return np.array([col,row],dtype=float)
        center=grid(p);scale=width/8.
        matrix=np.array([[scale*self.mpp,0,width/2-scale*self.mpp*center[0]],
                         [0,scale*self.mpp,(height-86)/2-scale*self.mpp*center[1]]],dtype=float)
        map_height=height-86
        view=cv2.warpAffine(self.native_map,matrix,(width,map_height),flags=cv2.INTER_NEAREST,borderValue=BG)
        def point(position):return tuple(np.rint(matrix[:,:2]@grid(position)+matrix[:,2]).astype(int))
        camera=np.asarray(frame['camera_position']);yaw=frame['camera_yaw']
        wedge=[point(camera)]
        for angle in np.linspace(yaw-np.pi/4,yaw+np.pi/4,30):
            wedge.append(point(camera+10*np.array([np.cos(angle),0,-np.sin(angle)])))
        tint=view.copy();cv2.fillPoly(tint,[np.asarray(wedge,dtype=np.int32)],YELLOW)
        view=cv2.addWeighted(view,.90,tint,.10,0)
        if plan:
            for candidate in plan['candidates']:
                trajectory=np.asarray(candidate['query']['nominal_timed_trajectory'])
                pts=np.asarray([point(pos) for pos in trajectory[:,1:4]],dtype=np.int32)
                cv2.polylines(view,[pts],False,GRAY,1,cv2.LINE_AA)
            selected=plan['candidates'][plan['selected']]['query']
            pts=np.asarray([point(pos) for pos in np.asarray(selected['nominal_timed_trajectory'])[:,1:4]],dtype=np.int32)
            cv2.polylines(view,[pts],False,CYAN,4,cv2.LINE_AA)
        executed=np.asarray([point(f['robot_position']) for f in self.frames[:index+1]],dtype=np.int32)
        cv2.polylines(view,[executed],False,WHITE,2,cv2.LINE_AA)
        for track in frame['human_tracks']:
            if not track['valid']:continue
            xy=point(track['position']);visible=track['visible_now']
            cv2.circle(view,xy,7,YELLOW,-1 if visible else 2,cv2.LINE_AA)
            if not visible:text(view,f'{track["observation_age"]:.1f}s',(xy[0]+8,xy[1]-10),13,YELLOW)
        if self.args.show_ground_truth:
            for human in self.truth[round(frame['timestamp'],8)]:
                cv2.drawMarker(view,point(human['witness']['torso']),MAGENTA,cv2.MARKER_TILTED_CROSS,15,2)
        goal=point(self.goal);clamped=(int(np.clip(goal[0],18,width-18)),int(np.clip(goal[1],18,map_height-18)))
        cv2.drawMarker(view,clamped,GREEN,cv2.MARKER_STAR,18,2,cv2.LINE_AA)
        text(view,'GOAL' if goal==clamped else 'GOAL off-map',
             (int(np.clip(clamped[0]-45,5,width-105)),int(np.clip(clamped[1]+10,5,map_height-22))),13,GREEN)
        robot=point(p);direction=point(p+.45*np.array([np.cos(frame['robot_yaw']),0,-np.sin(frame['robot_yaw'])]))
        cv2.circle(view,robot,8,BLUE,-1,cv2.LINE_AA);cv2.arrowedLine(view,robot,direction,BLUE,3,cv2.LINE_AA,tipLength=.45)
        canvas[40:40+map_height]=view
        text(canvas,'Cyan selected | gray candidates | white executed',(12,height-42),15)
        text(canvas,'Blue robot | green goal | yellow visible / hollow remembered',(12,height-23),14)
        if self.args.show_ground_truth:text(canvas,'Magenta crosses = privileged current GT (diagnostic only)',(12,40),13,MAGENTA)
        return canvas

    def telemetry(self,index,width=512,height=414):
        frame=self.frames[index];plan=self.plan(index);panel=np.full((height,width,3),BG,np.uint8)
        action=self.actions[index-1] if index else None
        distance=action['goal_distance'] if action else self.metrics['initial_goal_distance_m']
        progress=100*(1-distance/self.metrics['initial_goal_distance_m'])
        selected=plan['candidates'][plan['selected']] if plan else None
        text(panel,self.policy.upper()+'  |  recorded policy telemetry',(14,8),20,BLUE if self.policy=='static' else CYAN)
        text(panel,f'Sim {frame["timestamp"]:.3f}s   Goal {distance:.2f}m   Progress {progress:.1f}%',(14,40),17)
        family=selected['family'] if selected else 'initial observation WAIT'
        text(panel,f'Selected {family}   |   action #{index}',(14,65),18,CYAN)
        v,w=action['command_native'] if action else (0,0)
        speed=v*self.clock['integration_dt']/self.clock['control_dt']
        angular=w*self.clock['integration_dt']/self.clock['control_dt']
        state='STOP' if action and action['action']=='agent_0_discrete_stop' else 'WAIT' if abs(speed)<.001 and abs(angular)<.001 else 'TURN' if abs(speed)<.001 else 'SLOW' if speed<.5 else 'MOVE'
        text(panel,f'{state}   Last command {speed:.2f}m/s   Yaw {angular:.2f}rad/s',(14,90),16)
        visible=sum(t['visible_now'] for t in frame['human_tracks']);remembered=sum(t['valid'] and not t['visible_now'] for t in frame['human_tracks'])
        history=sum(t>=frame['timestamp']-8 for t in self.times[:index+1])
        text(panel,f'History {history} frames | visible {visible} | remembered {remembered}',(14,115),16)
        latency=1000*plan['latency']['inference_s'] if plan else 0.
        text(panel,f'WM inference {latency:.2f}ms at plan t={plan["timestamp"]:.2f}s' if plan else 'WM inference: no decision yet',(14,140),16)
        text(panel,'Full timed-query collision risk (recorded predictions)',(14,165),15,MUTED)
        if plan:
            for i,(candidate,safe) in enumerate(zip(plan['candidates'],plan['safe_probabilities'])):
                y=189+i*19
                selected_row=i==plan['selected']
                if selected_row:cv2.rectangle(panel,(8,y-1),(width-10,y+19),(32,64,72),-1)
                text(panel,('> ' if selected_row else '  ')+candidate['family'],(12,y),15,CYAN if selected_row else FG)
                if safe is None:
                    text(panel,'WM disabled',(245,y),15,MUTED)
                else:
                    risk=1-safe;color=GREEN if risk<=.2 else YELLOW if risk<=.4 else RED
                    cv2.rectangle(panel,(212,y+5),(392,y+14),(46,55,69),-1)
                    cv2.rectangle(panel,(212,y+5),(212+int(180*risk),y+14),color,-1)
                    text(panel,f'{100*risk:5.1f}%',(408,y),15,color)
        status,color=self.status(index)
        final=f'  |  SPL {self.metrics["spl"]:.2f}  PSC {self.metrics["psc"]:.3f}' if index==len(self.frames)-1 else ''
        text(panel,status+final,(14,height-29),17,color)
        return panel

    def diagnostic(self,index):
        photo=cv2.resize(self.rgb(index),(768,768),interpolation=cv2.INTER_LINEAR)
        overlay=photo.copy();cv2.rectangle(overlay,(0,0),(768,68),BG,-1);photo=cv2.addWeighted(photo,.15,overlay,.85,0)
        text(photo,self.policy.upper()+'  •  native Falcon third-person RGB',(18,10),25)
        text(photo,self.tag+'  |  validated recorded-action replay',(18,43),17)
        jaw=cv2.resize(self.rgb(index,'jaw'),(224,224))
        photo[502:726,16:240]=jaw;cv2.rectangle(photo,(15,501),(241,727),WHITE,2)
        text(photo,'Jaw RGB • diagnostic',(20,733),17)
        status,color=self.status(index)
        if index==len(self.frames)-1:
            cv2.rectangle(photo,(256,690),(753,758),BG,-1);text(photo,status,(277,710),28,color)
        right=np.concatenate([self.local_map(index),self.telemetry(index)],axis=0)
        return np.concatenate([photo,right],axis=1)

    def comparison_half(self,index,sync_time):
        panel=np.full((768,640,3),BG,np.uint8);panel[52:692]=self.rgb(index)
        text(panel,self.policy.upper()+'  |  native Falcon RGB',(14,7),24,BLUE if self.policy=='static' else CYAN)
        text(panel,f'Common simulation clock {sync_time:.3f}s',(14,35),14)
        plan=self.plan(index);family=plan['candidates'][plan['selected']]['family'] if plan else 'initial WAIT'
        action=self.actions[index-1] if index else None;distance=action['goal_distance'] if action else self.metrics['initial_goal_distance_m']
        status,color=self.status(index)
        text(panel,f'Native t={self.frames[index]["timestamp"]:.3f}s | {family} | goal {distance:.2f}m',(12,698),17)
        held=' (terminal frame held)' if sync_time>self.times[-1]+1e-6 else ''
        text(panel,status+(f'  |  PSC {self.metrics["psc"]:.3f}' if index==len(self.frames)-1 else ''),(12,727),19,color)
        if held:text(panel,held.strip(),(12,751),12,color)
        local=self.local_map(index)[40:-46]
        panel[539:665,384:624]=cv2.resize(local,(240,126),interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel,(383,538),(625,666),WHITE,1)
        text(panel,'LOCAL PLAN / EXECUTED TRAJECTORY',(385,671),11)
        return panel


def compose(args):
    from habitat.utils.visualizations.utils import images_to_video
    from eventwm_closed_loop import save,research_lock,research_guard
    cv2.setNumThreads(1);lock=research_lock(args.output);guard=research_guard(args.output);guard.check()
    videos=args.output/'videos';screens=args.output/'screenshots';screens.mkdir(exist_ok=True)
    success=EpisodeView(args,'eventwm','ep001_YRUkbU5xsYj_6')
    collision=EpisodeView(args,'eventwm','ep004_bCPU9suPUw9_10')
    static=EpisodeView(args,'static','ep004_bCPU9suPUw9_10')
    for key in ('seed','start_position','start_yaw','goal','scene_id','episode_id'):
        assert collision.metrics[key]==static.metrics[key],('Paired initialization differs',key)
    first_change=next((a['timestamp'] for a,b in zip(static.actions,collision.actions)
        if not np.allclose(a['command_native'],b['command_native'],rtol=1e-5,atol=1e-5)),None)
    artifacts={}
    for name,view in (('A_eventwm_success',success),('B_eventwm_collision',collision)):
        frames=[]
        for index in range(len(view.frames)):
            if (args.output/'STOP_RESEARCH').exists():raise RuntimeError('Stopped visualization only')
            guard.check();frames.append(view.diagnostic(index))
        frames.extend([frames[-1]]*24)
        images_to_video(frames,str(videos),name,fps=12,quality=8,verbose=False,ffmpeg_params=['-threads','1'],codec='libx264')
        import imageio.v2 as imageio
        representative=len(view.frames)//2 if name.startswith('A') else len(view.frames)-1
        imageio.imwrite(screens/(name+'.png'),frames[representative])
        artifacts[name]=dict(video=str(videos/(name+'.mp4')),screenshot=str(screens/(name+'.png')),
                             validation=str(view.directory/'validation.json'))
    frames=[];dt=.048;end=max(static.times[-1],collision.times[-1])
    timeline=np.r_[np.arange(0,end,dt),end]
    for t in timeline:
        if (args.output/'STOP_RESEARCH').exists():raise RuntimeError('Stopped visualization only')
        guard.check();left=static.comparison_half(static.frame_index(t),t);right=collision.comparison_half(collision.frame_index(t),t)
        if first_change is not None and t>=first_change:
            text(left,f'Actions diverged @ {first_change:.3f}s',(343,34),14,YELLOW)
            text(right,f'Actions diverged @ {first_change:.3f}s',(343,34),14,YELLOW)
        frames.append(np.concatenate([left,right],axis=1))
    frames.extend([frames[-1]]*24)
    name='C_static_vs_eventwm'
    images_to_video(frames,str(videos),name,fps=12,quality=8,verbose=False,ffmpeg_params=['-threads','1'],codec='libx264')
    imageio.imwrite(screens/(name+'.png'),frames[int(np.searchsorted(timeline,collision.times[-1],side='left'))])
    imageio.imwrite(screens/(name+'_final.png'),frames[-1])
    artifacts[name]=dict(video=str(videos/(name+'.mp4')),screenshot=str(screens/(name+'.png')),
        common_simulation_step_s=dt,first_divergent_robot_action_time_s=first_change,
        selection='Latest actual native frame at or before common simulation time',
        after_terminal='Hold final frame with explicit terminal label; no fabricated continuing motion')
    save(args.output/'video_manifest.json',dict(artifacts=artifacts,paired_initialization_verified=True,
        policy_unchanged=True,metrics_unchanged=True,ground_truth_map_overlay=args.show_ground_truth,
        playback='12 fps; native-action frames preserve simulation timestamps; slower than native time; final two-second hold',
        native_utilities=['Habitat observations_to_image','Habitat get_topdown_map/colorize_topdown_map/to_grid','Habitat images_to_video'],
        telemetry='Original model predictions and inference latency; no new model inference in visualization process'))
    print(json.dumps(artifacts,indent=2))
