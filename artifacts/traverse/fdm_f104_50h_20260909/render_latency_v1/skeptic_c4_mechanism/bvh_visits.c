/* Count ray-traversal work for the Mesa 24.2.8 lavapipe BVH layout (index-midpoint split, leaves in submitted order,
   2 children with stored child boxes) under two child-visit orders: 0 = child 0 always first (24.2.8's no-op "sort"),
   1 = nearer child first (25.2.8's fixed sort). Rays: overhead pinhole camera at (0,0,110), hfov 47 deg, 1024x1024
   grid subsampled every STEP pixels, max distance 180 m. Terrain only (no vehicle). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
typedef struct { float mn[3], mx[3]; } aabb;
typedef struct { int child[2]; aabb b[2]; } node;
static float *T; static node *nodes; static int nnodes = 0;
#define INVALID 0x7fffffff
static void tri_box(int i, aabb *a) { for (int k=0;k<3;k++){ float x=T[9*i+k],y=T[9*i+3+k],z=T[9*i+6+k];
  a->mn[k]=fmin(x,fmin(y,z)); a->mx[k]=fmax(x,fmax(y,z)); } }
static int build(int first, int last) {
  int id = nnodes++; node *n = &nodes[id]; int split = (int)(((unsigned)first + (unsigned)last) / 2);
  int c0, c1;
  if (first < split) c0 = build(first, split); else c0 = -(first + 1);
  if (first < last) { if (split + 1 < last) c1 = build(split + 1, last); else c1 = -(last + 1); } else c1 = INVALID;
  n = &nodes[id]; n->child[0] = c0; n->child[1] = c1;
  for (int i=0;i<2;i++){ int c=n->child[i]; aabb *a=&n->b[i];
    if (c==INVALID){ for(int k=0;k<3;k++){a->mn[k]=INFINITY;a->mx[k]=-INFINITY;} continue; }
    if (c<0) tri_box(-c-1,a); else { node *ch=&nodes[c]; for(int k=0;k<3;k++){a->mn[k]=fmin(ch->b[0].mn[k],ch->b[1].mn[k]); a->mx[k]=fmax(ch->b[0].mx[k],ch->b[1].mx[k]);} } }
  return id;
}
static int tri_hit(int i, const double *o, const double *d, double *t) {
  double v0[3],e1[3],e2[3],p[3],q[3],s[3];
  for(int k=0;k<3;k++){v0[k]=T[9*i+k]; e1[k]=T[9*i+3+k]-v0[k]; e2[k]=T[9*i+6+k]-v0[k];}
  p[0]=d[1]*e2[2]-d[2]*e2[1]; p[1]=d[2]*e2[0]-d[0]*e2[2]; p[2]=d[0]*e2[1]-d[1]*e2[0];
  double det=e1[0]*p[0]+e1[1]*p[1]+e1[2]*p[2]; if (fabs(det)<1e-12) return 0; double inv=1.0/det;
  for(int k=0;k<3;k++) s[k]=o[k]-v0[k]; double u=(s[0]*p[0]+s[1]*p[1]+s[2]*p[2])*inv; if(u<0||u>1) return 0;
  q[0]=s[1]*e1[2]-s[2]*e1[1]; q[1]=s[2]*e1[0]-s[0]*e1[2]; q[2]=s[0]*e1[1]-s[1]*e1[0];
  double v=(d[0]*q[0]+d[1]*q[1]+d[2]*q[2])*inv; if(v<0||u+v>1) return 0;
  double tt=(e2[0]*q[0]+e2[1]*q[1]+e2[2]*q[2])*inv; if (tt<=0) return 0; *t=tt; return 1;
}
int main(int argc, char **argv) {
  const char *path = argv[1]; int order = atoi(argv[2]); int STEP = atoi(argv[3]);
  FILE *f = fopen(path, "rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
  int N = (int)(sz / 36); T = malloc(sz); if (fread(T,1,sz,f)!=(size_t)sz) return 1; fclose(f);
  nodes = malloc(sizeof(node) * (size_t)N); int root = build(0, N - 1);
  const int W=1024; double fpx=(W/2.0)/tan(47.0*M_PI/180.0/2.0); double o[3]={0,0,110};
  long rays=0, inner=0, leaves=0, boxtests=0, hits=0; double tsum=0; int stack[256];
  for (int v=0; v<W; v+=STEP) for (int u=0; u<W; u+=STEP) {
    double d[3]={(u-(W-1)/2.0)/fpx, -(v-(W-1)/2.0)/fpx, -1.0}; double nrm=sqrt(d[0]*d[0]+d[1]*d[1]+1); for(int k=0;k<3;k++) d[k]/=nrm;
    double inv[3]; for(int k=0;k<3;k++) inv[k]= d[k]==0? FLT_MAX : 1.0/d[k];
    double tmax=180.0; int sp=0; int cur=root; int hit=0; rays++;
    for(;;){
      if (cur==INVALID){ if(sp==0) break; cur=stack[--sp]; }
      if (cur<0){ leaves++; double t; if(tri_hit(-cur-1,o,d,&t) && t<tmax){tmax=t;hit=1;} cur=INVALID; continue; }
      inner++; node *n=&nodes[cur]; int ci[2]={INVALID,INVALID}; double dist[2]={INFINITY,INFINITY};
      for(int i=0;i<2;i++){ boxtests++; aabb *a=&n->b[i]; if(!(a->mn[0]==a->mn[0])) continue;
        double tmn=-INFINITY,tmx=INFINITY; for(int k=0;k<3;k++){ double b0=(a->mn[k]-o[k])*inv[k], b1=(a->mx[k]-o[k])*inv[k];
          tmn=fmax(tmn,fmin(b0,b1)); tmx=fmin(tmx,fmax(b0,b1)); }
        if (n->child[i]!=INVALID && tmx>=fmax(0.0,tmn) && tmn<tmax){ ci[i]=n->child[i]; dist[i]=tmn; } }
      if (order==1 && dist[1]<dist[0]){ int t=ci[0]; ci[0]=ci[1]; ci[1]=t; }
      cur=ci[0]; if (ci[1]!=INVALID){ if(sp>=256){fprintf(stderr,"stack overflow\n");return 2;} stack[sp++]=ci[1]; }
    }
    if (hit){hits++; tsum+=tmax;}
  }
  printf("{\"file\":\"%s\",\"order\":\"%s\",\"triangles\":%d,\"rays\":%ld,\"hit_rays\":%ld,\"mean_hit_t\":%.6f,"
         "\"inner_nodes_per_ray\":%.2f,\"triangle_tests_per_ray\":%.2f}\n", path, order?"nearer_child_first":"child0_first",
         N, rays, hits, tsum/(hits?hits:1), (double)inner/rays, (double)leaves/rays);
  return 0;
}
