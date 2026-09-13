"""Continuous arc-length lookahead expert; no simulator dependencies."""
import bisect
import math


class ContinuousRouteExpert:
    def __init__(self, path, goal, lookahead=.5):
        self.path=[]
        for p in path:
            p=list(map(float,p))
            if len(p)!=3 or not all(map(math.isfinite,p)):
                raise ValueError('finite XYZ path required')
            if not self.path or math.dist(p[:2],self.path[-1][:2])>1e-6:
                self.path.append(p)
        self.goal=list(map(float,goal))
        if not self.path or len(self.goal)!=3 or not all(map(math.isfinite,self.goal)):
            raise ValueError('nonempty path and finite XYZ goal required')
        if math.dist(self.path[-1][:2],self.goal[:2])>1e-6:
            self.path.append(self.goal)
        if not math.isfinite(lookahead) or lookahead<=0:
            raise ValueError('positive lookahead required')
        self.lookahead=lookahead
        self.arc=[0.]
        for a,b in zip(self.path,self.path[1:]):
            self.arc.append(self.arc[-1]+math.dist(a[:2],b[:2]))
        self.progress=0.;self.index=0

    def metadata(self):
        return dict(type='continuous_arc_lookahead_v1',lookahead_m=self.lookahead,
                    projection_forward_window_m=.6,max_forward_mps=.35,
                    heading_gain=1.5,heading_soft_gate_rad=1.2,
                    terminal_speed_gain=.6,terminal_speed_floor_mps=.08,
                    stop_radius_m=.30,terminal_progress_tolerance_m=.35)

    def point(self,s):
        s=max(0.,min(s,self.arc[-1]))
        i=min(bisect.bisect_right(self.arc,s)-1,len(self.path)-2)
        if len(self.path)==1:return self.path[0][:]
        f=(s-self.arc[i])/(self.arc[i+1]-self.arc[i])
        return [a+f*(b-a) for a,b in zip(self.path[i],self.path[i+1])]

    def command(self,state):
        p=state['position_w'];w,x,y,z=state['quaternion_wxyz']
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        # Search only a short connected arc interval: no global nearest-point
        # jumps at crossings or nearby parallel/reversed route sections.
        lo=max(0.,self.progress-.1);hi=min(self.arc[-1],self.progress+.6)
        best=(float('inf'),self.progress)
        for i,(a,b) in enumerate(zip(self.path,self.path[1:])):
            left=max(lo,self.arc[i]);right=min(hi,self.arc[i+1])
            if left>right:continue
            length=self.arc[i+1]-self.arc[i]
            f=sum((p[k]-a[k])*(b[k]-a[k]) for k in (0,1))/(length*length)
            s=max(left,min(right,self.arc[i]+f*length))
            q=self.point(s);candidate=(math.dist(p[:2],q[:2]),s)
            if candidate<best:best=candidate
        self.progress=max(self.progress,best[1])
        remaining=self.arc[-1]-self.progress
        terminal=remaining<=.35
        self.index=len(self.path)-1 if terminal else min(bisect.bisect_right(self.arc,self.progress),len(self.path)-2)
        target=self.point(self.progress+self.lookahead)
        dx,dy=target[0]-p[0],target[1]-p[1]
        error=math.atan2(math.sin(math.atan2(dy,dx)-yaw),math.cos(math.atan2(dy,dx)-yaw))
        distance=math.dist(p[:2],self.goal[:2])
        stop=terminal and distance<=.30
        # C1 heading gate avoids a discontinuous forward/turn threshold.
        gate=math.cos(min(abs(error)/1.2,1.)*math.pi/2)**2
        cruise=.35/(1.+.8*abs(error))
        braking=min(.35,max(.08,.6*max(remaining,distance)))
        raw=[0.,0.,0.] if stop else [min(cruise,braking)*gate,0.,1.5*error]
        return raw,dict(waypoint_index=self.index,waypoint_count=len(self.path),waypoint=target,
            waypoint_distance_xy_m=math.hypot(dx,dy),heading_error_rad=error,
            goal_distance_xy_m=distance,stop_intent=stop,
            progress_m=self.progress,remaining_path_m=remaining,
            cross_track_distance_m=math.dist(p[:2],self.point(self.progress)[:2]),
            phase='stop' if stop else ('turn' if abs(error)>=1.2 else 'track'))
