#!/bin/bash
# My own billed node-hours on the AMD cluster since the start of this effort (2026-09-21 13:00), by partition, using the
# TRESBillingWeights per raw node-hour: mi2101x 0.1, mi3501x 0.125, mi3001x 0.125, mi2104x 0.4, mi2508x 0.8, mi3008x 1.0, mi3508x 1.2, devel 0.
ssh amd 'sacct -u $USER -S 2026-09-21T13:00 -X -n -o Partition,ElapsedRaw,NNodes,State -P' | awk -F'|' '
BEGIN{w["mi2101x"]=0.1;w["mi3501x"]=0.125;w["mi3001x"]=0.125;w["mi2104x"]=0.4;w["mi2508x"]=0.8;w["mi3008x"]=1.0;w["mi3508x"]=1.2;w["mi3258x"]=1.2;w["devel"]=0}
{p=$1; h=$2/3600*$3; raw[p]+=h; bill[p]+=h*(p in w?w[p]:1.0); tot+=h*(p in w?w[p]:1.0)}
END{for(p in raw) printf "%-9s raw %7.2f h  billed %7.2f\n", p, raw[p], bill[p]; printf "TOTAL billed %.2f node-hours (cap 100)\n", tot}'
