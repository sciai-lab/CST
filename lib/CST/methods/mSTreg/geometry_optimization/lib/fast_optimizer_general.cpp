/*
compile with the following two commands from command line (Linux only):
g++ -c -Wall -Werror -O3 -fpic fast_optimizer_general.cpp && g++ -shared -Wall -O3 -o fast_optimizer_general.so fast_optimizer_general.o
The flags -Wall -O3 can technically be omitted, but should not!
*/

#include <math.h>
#include <iostream>
#include <queue>
#include <cstdlib>
using namespace std;


void optimize_stack(int DIMENSION, int NUMSITES,int NUM_BPS, int NUM_EDGES,int max_num_neighs,int indices[], int indptr[],double EW[], double EL[], double XX[],double al){
    double tol = 1e-10;

    double B[NUM_EDGES],C[NUM_BPS][DIMENSION];
    int eqnstack[NUMSITES], leafQ[NUM_BPS], val[NUM_BPS], intern_edge[NUM_EDGES];
    int m;
    int lqp,eqp;
    int k;
    double q0,q1;
    double suma,t;
    double q[max_num_neighs];
    int BP_idx,BP_idx_glob,node,neigh_idx_glob,neigh_idx,edge_index,edge_index2,edge_index3;


    lqp = eqp = 0;
    /* prepare equations */
    for(BP_idx=NUM_BPS-1;BP_idx>=0;BP_idx--){
        k=0;
        suma=0;
        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++){
            q[k]=pow(fabs(EW[edge_index]),al)/(EL[edge_index]+tol);
            suma+=q[k];

            k++;
        }
//         //TODO: Remove this
//        ///////////////////////////////////////////////////////////////////////
//        std::cout<< "BP_idx: " << BP_idx << std::endl;
//        std::cout<< "indptr[BP_idx]"<< indptr[BP_idx] << std::endl;
//        std::cout<< "indptr[BP_idx+1]"<< indptr[BP_idx+1] << std::endl;
//        std::cout << "suma: " << suma << std::endl;
//        // print q[0:k]
//        for (int i=0; i<k; i++){
//            if (i==0)
//            {
//                std::cout << "q: ";
//            }
//            std::cout << q[i] << " ";
//        }
//        std::cout << std::endl;
//
//        // print EW[indptr[BP_idx]:indptr[BP_idx+1]]
//        for (int i=indptr[BP_idx]; i<indptr[BP_idx+1]; i++){
//            if (i==indptr[BP_idx]){
//                std::cout << "EW: ";
//            }
//            std::cout << EW[i] << " ";
//        }
//        std::cout << std::endl;
//
//        // print EL[indptr[BP_idx]:indptr[BP_idx+1]]
//        for (int i=indptr[BP_idx]; i<indptr[BP_idx+1]; i++){
//            if (i==indptr[BP_idx]){
//                std::cout << "EL: ";
//            }
//            std::cout << EL[i] << " ";
//        }
//        std::cout << std::endl;
        //////////////////////////////////////////////////////////////////////////

        k=0;
        val[BP_idx] = 0;
        for(m=0;m<DIMENSION;m++){C[BP_idx][m] = 0.0;}

        #define prep(a,b,c) if(b>=NUMSITES){val[BP_idx]++;B[a]=c;intern_edge[a]=1;}else{for(m=0;m<DIMENSION;m++){C[BP_idx][m]+=XX[b*DIMENSION+m]*c;}}
        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++){
            q[k]/=suma;

            B[edge_index] = 0.0;
            intern_edge[edge_index] = 0;

            node=indices[edge_index];
            prep(edge_index,node,q[k]);


            //TODO: Remove this
//            /////////////////////////////////////////////////////////////////
//            std::cout << "edge_index=" << edge_index << std::endl;
//            std::cout << "node=" << node << std::endl;
//            std::cout << "q[k]=" << q[k] << std::endl;
//
//            // print C[BP_idx]
//            for (int i=0; i<DIMENSION; i++){
//                if (i==0){
//                    std::cout << "INIT C["<<BP_idx<<"]=: ";
//                }
//                std::cout << C[BP_idx][i] << " ";
//            }
//            std::cout<<std::endl;
//            for (int i=0; i<DIMENSION; i++){
//                if (i==0){
//                    std::cout << "INIT XX["<<node<<"]=: ";
//                }
//                std::cout << XX[node*DIMENSION+i] << " ";
//            }
//            std::cout<<std::endl;
//            /////////////////////////////////////////////////////////////////
            k++;
        }

        if(val[BP_idx]<=1){leafQ[lqp]=BP_idx,lqp++;}
    }
    while(lqp > 1){

        /* eliminate leaf i from tree*/
        lqp--; BP_idx = leafQ[lqp]; val[BP_idx]--; BP_idx_glob = BP_idx + NUMSITES;
        eqnstack[eqp] = BP_idx; eqp++;/* push BP_idx in stack */
        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++){
            if(intern_edge[edge_index] != 0){break;}
        }

        q0 = B[edge_index];
        neigh_idx = indices[edge_index]-NUMSITES;/* neighbor is j */
        val[neigh_idx]-- ;
        if(val[neigh_idx] == 1){ leafQ[lqp] = neigh_idx; lqp ++; }/* check if neighbor has become leaf? */

        for (edge_index2=indptr[neigh_idx];edge_index2<indptr[neigh_idx+1];edge_index2++){
            if(indices[edge_index2]==BP_idx_glob){break;}
        }
        q1 = B[edge_index2]; B[edge_index2] = 0.0; intern_edge[edge_index2] = 0;
        t = 1.0-q1*q0; t = 1.0/t;
        for (edge_index3=indptr[neigh_idx];edge_index3<indptr[neigh_idx+1];edge_index3++){
            B[edge_index3] *= t;
        }

        for(m=0; m<DIMENSION; m++){ C[neigh_idx][m] += q1*C[BP_idx][m]; C[neigh_idx][m] *= t; }
        //TODO: Remove this
//        t = (1-q0);
//        for (edge_index3=indptr[neigh_idx];edge_index3<indptr[neigh_idx+1];edge_index3++){
//            B[edge_index3] *= t;
//        }
//
//        for(m=0; m<DIMENSION; m++){C[neigh_idx][m] *= t; C[neigh_idx][m] -= q1*C[BP_idx][m];  }
    }
    /* Solve trivial tree */
    BP_idx = leafQ[0]; BP_idx_glob = BP_idx + NUMSITES;
    for(m=0; m <DIMENSION; m++){XX[BP_idx_glob*DIMENSION+m] = C[BP_idx][m];}
    /* Solve rest by backsolving */
    while(eqp > 0){
        eqp--; BP_idx = eqnstack[eqp]; BP_idx_glob = BP_idx + NUMSITES;
        /////////////////////////////////////////////////////
//        //TODO: remove this
//        //print q
//        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++){
//            if (edge_index==indptr[BP_idx])
//            {
//                std::cout << "B: ";
//            }
//            std::cout << B[edge_index] << " ";
//        }
//        std::cout << std::endl;
//
//        // print intern edges
//        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++){
//            if (edge_index==indptr[BP_idx])
//            {
//                std::cout << "intern_edge: ";
//            }
//            std::cout << intern_edge[edge_index] << " ";
//        }
//        std::cout << std::endl;
//
//        int cacho=0;
//        /////////////////////////////////////////////////////////////////
        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++)
        {
            if(intern_edge[edge_index] != 0){break;}/* find edge index */
//            cacho++;
        }
        q0 = B[edge_index];
        neigh_idx_glob = indices[edge_index];/* get neighbor index */
//        // TODO: Remove this
//        //////////////////////////////////////////////////////////////////////////
//        std::cout <<"q0="<<q0<<std::endl;
//        std::cout <<"cacho="<<cacho<<std::endl;
        // print C[BP_idx]
//        for (int i=0; i<DIMENSION; i++){
//            if (i==0){
//                std::cout << "C["<<BP_idx<<"]=: ";
//            }
//            std::cout << C[BP_idx][i] <<std::endl;
//        }
        //////////////////////////////////////////////////////////////////////////
        for(m = 0; m < DIMENSION; m++){XX[BP_idx_glob*DIMENSION+m] = C[BP_idx][m] + q0*XX[neigh_idx_glob*DIMENSION+m];}
    }
    return;
}




//When arrays are large, the stack can overflow. In this case, we use a heap instead.
void optimize_heap(int DIMENSION, int NUMSITES,int NUM_BPS, int NUM_EDGES,int max_num_neighs,int indices[], int indptr[],double EW[],double EL[], double XX[], double al) {
    double tol = 1e-10;

    double** C = new double*[NUMSITES];
    for (int i = 0; i < NUMSITES; ++i) {
        C[i] = new double[DIMENSION];
    }
    double* B = new double[NUM_EDGES];
    int* intern_edge = new int[NUM_EDGES];



    int eqnstack[NUMSITES], leafQ[NUM_BPS], val[NUM_BPS];
    int m;
    int lqp,eqp;
    int k;
    double q0,q1;
    double suma,t;
    double q[max_num_neighs];
    int BP_idx,BP_idx_glob,node,neigh_idx_glob,neigh_idx,edge_index,edge_index2,edge_index3;


    lqp = eqp = 0;
    /* prepare equations */
    for(BP_idx=NUM_BPS-1;BP_idx>=0;BP_idx--){
        k=0;
        suma=0;
        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++){
            q[k]=pow(fabs(EW[edge_index]),al)/(EL[edge_index]+tol);
            suma+=q[k];

            k++;
        }
//         //TODO: Remove this
//        ///////////////////////////////////////////////////////////////////////
//        std::cout<< "BP_idx: " << BP_idx << std::endl;
//        std::cout<< "indptr[BP_idx]"<< indptr[BP_idx] << std::endl;
//        std::cout<< "indptr[BP_idx+1]"<< indptr[BP_idx+1] << std::endl;
//        std::cout << "suma: " << suma << std::endl;
//        // print q[0:k]
//        for (int i=0; i<k; i++){
//            if (i==0)
//            {
//                std::cout << "q: ";
//            }
//            std::cout << q[i] << " ";
//        }
//        std::cout << std::endl;
//
//        // print EW[indptr[BP_idx]:indptr[BP_idx+1]]
//        for (int i=indptr[BP_idx]; i<indptr[BP_idx+1]; i++){
//            if (i==indptr[BP_idx]){
//                std::cout << "EW: ";
//            }
//            std::cout << EW[i] << " ";
//        }
//        std::cout << std::endl;
//
//        // print EL[indptr[BP_idx]:indptr[BP_idx+1]]
//        for (int i=indptr[BP_idx]; i<indptr[BP_idx+1]; i++){
//            if (i==indptr[BP_idx]){
//                std::cout << "EL: ";
//            }
//            std::cout << EL[i] << " ";
//        }
//        std::cout << std::endl;
        //////////////////////////////////////////////////////////////////////////

        k=0;
        val[BP_idx] = 0;
        for(m=0;m<DIMENSION;m++){C[BP_idx][m] = 0.0;}

        #define prep(a,b,c) if(b>=NUMSITES){val[BP_idx]++;B[a]=c;intern_edge[a]=1;}else{for(m=0;m<DIMENSION;m++){C[BP_idx][m]+=XX[b*DIMENSION+m]*c;}}
        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++){
            q[k]/=suma;

            B[edge_index] = 0.0;
            intern_edge[edge_index] = 0;

            node=indices[edge_index];
            prep(edge_index,node,q[k]);


            //TODO: Remove this
//            /////////////////////////////////////////////////////////////////
//            std::cout << "edge_index=" << edge_index << std::endl;
//            std::cout << "node=" << node << std::endl;
//            std::cout << "q[k]=" << q[k] << std::endl;
//
//            // print C[BP_idx]
//            for (int i=0; i<DIMENSION; i++){
//                if (i==0){
//                    std::cout << "INIT C["<<BP_idx<<"]=: ";
//                }
//                std::cout << C[BP_idx][i] << " ";
//            }
//            std::cout<<std::endl;
//            for (int i=0; i<DIMENSION; i++){
//                if (i==0){
//                    std::cout << "INIT XX["<<node<<"]=: ";
//                }
//                std::cout << XX[node*DIMENSION+i] << " ";
//            }
//            std::cout<<std::endl;
//            /////////////////////////////////////////////////////////////////
            k++;
        }

        if(val[BP_idx]<=1){leafQ[lqp]=BP_idx,lqp++;}
    }
    while(lqp > 1){

        /* eliminate leaf i from tree*/
        lqp--; BP_idx = leafQ[lqp]; val[BP_idx]--; BP_idx_glob = BP_idx + NUMSITES;
        eqnstack[eqp] = BP_idx; eqp++;/* push BP_idx in stack */
        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++){
            if(intern_edge[edge_index] != 0){break;}
        }

        q0 = B[edge_index];
        neigh_idx = indices[edge_index]-NUMSITES;/* neighbor is j */
        val[neigh_idx]-- ;
        if(val[neigh_idx] == 1){ leafQ[lqp] = neigh_idx; lqp ++; }/* check if neighbor has become leaf? */

        for (edge_index2=indptr[neigh_idx];edge_index2<indptr[neigh_idx+1];edge_index2++){
            if(indices[edge_index2]==BP_idx_glob){break;}
        }
        q1 = B[edge_index2]; B[edge_index2] = 0.0; intern_edge[edge_index2] = 0;
        t = 1.0-q1*q0; t = 1.0/t;
        for (edge_index3=indptr[neigh_idx];edge_index3<indptr[neigh_idx+1];edge_index3++){
            B[edge_index3] *= t;
        }

        for(m=0; m<DIMENSION; m++){ C[neigh_idx][m] += q1*C[BP_idx][m]; C[neigh_idx][m] *= t; }
        //TODO: Remove this
//        t = (1-q0);
//        for (edge_index3=indptr[neigh_idx];edge_index3<indptr[neigh_idx+1];edge_index3++){
//            B[edge_index3] *= t;
//        }
//
//        for(m=0; m<DIMENSION; m++){C[neigh_idx][m] *= t; C[neigh_idx][m] -= q1*C[BP_idx][m];  }
    }
    /* Solve trivial tree */
    BP_idx = leafQ[0]; BP_idx_glob = BP_idx + NUMSITES;
    for(m=0; m <DIMENSION; m++){XX[BP_idx_glob*DIMENSION+m] = C[BP_idx][m];}
    /* Solve rest by backsolving */
    while(eqp > 0){
        eqp--; BP_idx = eqnstack[eqp]; BP_idx_glob = BP_idx + NUMSITES;
        /////////////////////////////////////////////////////
//        //TODO: remove this
//        //print q
//        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++){
//            if (edge_index==indptr[BP_idx])
//            {
//                std::cout << "B: ";
//            }
//            std::cout << B[edge_index] << " ";
//        }
//        std::cout << std::endl;
//
//        // print intern edges
//        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++){
//            if (edge_index==indptr[BP_idx])
//            {
//                std::cout << "intern_edge: ";
//            }
//            std::cout << intern_edge[edge_index] << " ";
//        }
//        std::cout << std::endl;
//
//        int cacho=0;
//        /////////////////////////////////////////////////////////////////
        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++)
        {
            if(intern_edge[edge_index] != 0){break;}/* find edge index */
//            cacho++;
        }
        q0 = B[edge_index];
        neigh_idx_glob = indices[edge_index];/* get neighbor index */
//        // TODO: Remove this
//        //////////////////////////////////////////////////////////////////////////
//        std::cout <<"q0="<<q0<<std::endl;
//        std::cout <<"cacho="<<cacho<<std::endl;
        // print C[BP_idx]
//        for (int i=0; i<DIMENSION; i++){
//            if (i==0){
//                std::cout << "C["<<BP_idx<<"]=: ";
//            }
//            std::cout << C[BP_idx][i] <<std::endl;
//        }
        //////////////////////////////////////////////////////////////////////////
        for(m = 0; m < DIMENSION; m++){XX[BP_idx_glob*DIMENSION+m] = C[BP_idx][m] + q0*XX[neigh_idx_glob*DIMENSION+m];}
    }
    return;
}

void optimize(int DIMENSION, int NUMSITES,int NUM_BPS, int NUM_EDGES,int max_num_neighs,int indices[], int indptr[],double EW[],double EL[], double XX[],double al){
    // Check the size and decide whether to use dynamic or static allocation
    bool useDynamicAllocation = (NUM_EDGES > 17000*50);
    if (useDynamicAllocation) {
        //When arrays are large, the stack can overflow. In this case, we use a heap instead.
        optimize_heap(DIMENSION, NUMSITES,NUM_BPS, NUM_EDGES, max_num_neighs, indices, indptr, EW,EL, XX, al);
    } else {
        optimize_stack(DIMENSION, NUMSITES,NUM_BPS, NUM_EDGES, max_num_neighs, indices, indptr, EW,EL, XX, al);
    }
return;
}

double length(int DIMENSION, int NUMSITES,int NUM_BPS,int indices[], int indptr[],double EW[], double EL[], double XX[], double al) {
    /*calculates the cost of the current configuration and stores edge lengths in EL*/
    #define dist(a,b) t=0.0;for(m=0;m<DIMENSION;m++){r=XX[a*DIMENSION+m]-XX[b*DIMENSION+m];t+=r*r;}t=sqrt(t);

    int BP_idx,BP_idx_glob,neighbor_BP,edge_index,edge_index2,m;
    double leng,t,r;
    leng = 0.0;
    for (BP_idx=0;BP_idx<NUM_BPS;BP_idx++)
    {
        BP_idx_glob=BP_idx+NUMSITES;
        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++)
        {
            neighbor_BP=indices[edge_index];
            if (neighbor_BP<BP_idx_glob)
            {
                dist(BP_idx_glob,neighbor_BP);
                EL[edge_index]=t;

                leng+=pow(fabs(EW[edge_index]),al)*t;
                if (neighbor_BP>=NUMSITES)
                {
                    for (edge_index2=indptr[neighbor_BP-NUMSITES];edge_index2<indptr[neighbor_BP-NUMSITES+1];edge_index2++)
                    {
                        if (indices[edge_index2]==BP_idx_glob)
                        {
                            EL[edge_index2]=t;
                            break;
                        }
                    }
                }
            }
        }
    }
    return leng;
}




extern "C"
void calculate_EW_BCST(int NUMSITES, int NUM_BPS,int NUM_EDGES, int indices[], int indptr[] ,double EW[], double demands[], double al){
    /*calculates the flows on all edges using the same tree elimination scheme as the equation solver in optimize*/
    int node;
    int leafQ[NUM_BPS],val[NUM_BPS];
    int lqp = 0;
    int done_edges[NUM_EDGES];
    double d[NUM_BPS];
    int BP_idx = 0,edge_index=0;
    int BP_idx2,BP_idx_glob;
//    TODO: Remove this
//    //////////////////////////////
//    //print indices
//    for (int i=0; i<NUM_EDGES; i++){
//        if (i==0){
//            std::cout << "indices: ";
//        }
//        std::cout << indices[i] << " ";
//    }
//    //////////////////////////////////////////////////
    for (BP_idx=0;BP_idx<NUM_BPS;BP_idx++)
    {
        val[BP_idx] = 0;
        d[BP_idx] = 0.0;
        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++)
        {
            done_edges[edge_index] = 0;
            node=indices[edge_index];
            if(node>=NUMSITES)
            {
                val[BP_idx]++;
            }
            else
            {
                d[BP_idx]+=fabs(demands[node]);
                EW[edge_index]=(1-fabs(demands[node]))*fabs(demands[node]);
                done_edges[edge_index]=1;
            }
        }
        if(val[BP_idx]==1)
        {
            leafQ[lqp]=BP_idx;
            lqp++;
        }
    }
    // TODO: Remove this
//    std::cout<<"lqp"<<lqp<<std::endl;
    while(lqp>1)
    {
        lqp--;
        BP_idx = leafQ[lqp]; BP_idx_glob=BP_idx+NUMSITES; val[BP_idx]--;

        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++)
        {   /*find parent BP*/
            if(done_edges[edge_index]==0){break;}
        }
        EW[edge_index] = fabs(d[BP_idx])*(1-fabs(d[BP_idx]));
        done_edges[edge_index] = 1;

        BP_idx2 = indices[edge_index]-NUMSITES;
        for (edge_index=indptr[BP_idx2];edge_index<indptr[BP_idx2+1];edge_index++)
        {
            if(indices[edge_index]==BP_idx_glob){break;}
        }
        EW[edge_index] = fabs(d[BP_idx])*(1-fabs(d[BP_idx]));
        done_edges[edge_index] = 1;
        d[BP_idx2]+=fabs(d[BP_idx]);
        val[BP_idx2]--;
        /* check if neighbor has become leaf? */
        if(val[BP_idx2] == 1)
        {
            leafQ[lqp] = BP_idx2;
            lqp ++;
        }
    }
    //T//TODO: Remove this
     ////////////////////////////////////
    // print EW values
//    for (int i=0; i<NUM_EDGES; i++){
//        if (i==0){
//            std::cout << "EW: ";
//        }
//        std::cout << EW[i] << " ";
//    }
    ////////////////////////////////////
}

extern "C"
void calculate_EW_BOT_leaves(int NUMSITES, int NUM_BPS,int NUM_EDGES, int indices[], int indptr[] ,double EW[], double demands[], double al){
    // ASSUMES TERMINALS ARE LEAVES
    /*calculates the flows on all edges using the same tree elimination scheme as the equation solver in optimize*/
    int node;
    int leafQ[NUM_BPS],val[NUM_BPS];
    int lqp = 0;
    int done_edges[NUM_EDGES];
    double d[NUM_BPS];
    int BP_idx = 0,edge_index=0;
    int BP_idx2,BP_idx_glob;

    for (BP_idx=0;BP_idx<NUM_BPS;BP_idx++)
    {
        val[BP_idx] = 0;
        d[BP_idx] = 0.0;
        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++)
        {
            done_edges[edge_index] = 0;
            node=indices[edge_index];
            if(node>=NUMSITES)
            {
                val[BP_idx]++;
            }
            else
            {
                d[BP_idx]+=demands[node];
                EW[edge_index]=fabs(demands[node]);
                done_edges[edge_index]=1;
            }
        }
        if(val[BP_idx]==1)
        {
            leafQ[lqp]=BP_idx;
            lqp++;
        }
    }
    while(lqp>1)
    {
        lqp--;
        BP_idx = leafQ[lqp]; BP_idx_glob=BP_idx+NUMSITES; val[BP_idx]--;

        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++)
        {   /*find parent BP*/
            if(done_edges[edge_index]==0){break;}
        }
        EW[edge_index] = fabs(d[BP_idx]);
        done_edges[edge_index] = 1;

        BP_idx2 = indices[edge_index]-NUMSITES;
        for (edge_index=indptr[BP_idx2];edge_index<indptr[BP_idx2+1];edge_index++)
        {
            if(indices[edge_index]==BP_idx_glob){break;}
        }
        EW[edge_index] = fabs(d[BP_idx]);
        done_edges[edge_index] = 1;
        d[BP_idx2]+=d[BP_idx];
        val[BP_idx2]--;
        /* check if neighbor has become leaf? */
        if(val[BP_idx2] == 1)
        {
            leafQ[lqp] = BP_idx2;
            lqp ++;
        }
    }
}


extern "C"
void calculate_EW_BOT(int NUMSITES, int NUM_BPS,int NUM_EDGES, int indices[], int indptr[] ,double EW[], double demands[],
                        double al=1){
     // ASSUMES INDICES AND INDPTR AND EW ALSO CONTAINS INFORMATION ABOUT THE EDGES OUTGOING FROM THE TERMINALS
    /*calculates the flows on all edges using the same tree elimination scheme as the equation solver in optimize*/

    int neigh,BP_idx_glob,BP_idx2,terminal,BP_idx2_glob;
    int BP_idx = 0,edge_index=0,edge_index2=0;
    int leafQ[NUM_BPS],val[NUM_BPS];
    int lqp = 0;
    int done_edges[NUM_EDGES];
    double d[NUM_BPS+NUMSITES];
    int degrees_terminals[NUMSITES];
    int max_deg=1;

    //init done_edges with zeros
    for (int i=0; i<NUM_EDGES; i++){
        done_edges[i]=0;
    }
    //init d with zeros
    for (int i=0; i<NUM_BPS+NUMSITES; i++){
        d[i]=0.0;
    }


    // initialize EW outgoing from terminals
    for (int terminal=0; terminal<NUMSITES; terminal++)
    {
        // Determine degree terminals
        degrees_terminals[terminal]=indptr[terminal+1]-indptr[terminal];
        if (degrees_terminals[terminal]>max_deg){max_deg=degrees_terminals[terminal];}

        d[terminal]=demands[terminal];
        if (degrees_terminals[terminal]==1)
        {
            edge_index=indptr[terminal];
            EW[edge_index]=fabs(d[terminal]);
            done_edges[edge_index]=1;
        }
        else
        {
            for(edge_index=indptr[terminal];edge_index<indptr[terminal+1];edge_index++){done_edges[edge_index]=0;}
        }
    }

    // initialize EW outgoing from BPs
    for (BP_idx_glob=NUMSITES;BP_idx_glob<NUMSITES+NUM_BPS;BP_idx_glob++)
    {
        BP_idx=BP_idx_glob-NUMSITES;
        val[BP_idx] = 0;
        d[BP_idx_glob] = 0.0;
        for (edge_index=indptr[BP_idx_glob];edge_index<indptr[BP_idx_glob+1];edge_index++)
        {
            done_edges[edge_index] = 0;
            neigh=indices[edge_index];

            if(neigh>=NUMSITES)
            {
                val[BP_idx]++;
            }
            else if (neigh<NUMSITES && degrees_terminals[neigh]>1)
            {
                val[BP_idx]++;
            }
            else
            {
                d[BP_idx_glob]+=d[neigh];
                EW[edge_index]=fabs(d[neigh]);
                done_edges[edge_index]=1;
            }
        }
        if(val[BP_idx]==1)
        {
            leafQ[lqp]=BP_idx;
            lqp++;
        }
    }

////    // TODO: Remove this
//    std::cout<<"lqp"<<lqp<<std::endl;
//    // print EW values
//    for (int i=0; i<NUM_EDGES; i++){
//        if (i==0){
//            std::cout << "EW: ";
//        }
//        std::cout << EW[i] << " ";
//    }
//    std::cout<<std::endl;
////    print done_edges
//    for (int i=0; i<NUM_EDGES; i++){
//        if (i==0){
//            std::cout << "done_edges: ";
//        }
//        std::cout << done_edges[i] << " ";
//    }std::cout<<std::endl;

    while(lqp>1)
    {
        lqp--;
        BP_idx = leafQ[lqp]; BP_idx_glob=BP_idx+NUMSITES; val[BP_idx]--;


        for (edge_index=indptr[BP_idx_glob];edge_index<indptr[BP_idx_glob+1];edge_index++)
        {   /*find parent BP*/
            if(done_edges[edge_index]==0){break;}
        }
        if(edge_index==indptr[BP_idx_glob+1]){continue;}// all adjacent edges have been updated already

        EW[edge_index] = fabs(d[BP_idx_glob]);
        done_edges[edge_index] = 1;

//        TODO: Remove this
//        std::cout<<"BP_idx="<<BP_idx+NUMSITES<<std::endl;
//        std::cout<<"edge_index="<<edge_index<<std::endl;
//        std::cout<<"indices[edge_index]="<<indices[edge_index]<<std::endl;
//        std::cout<<"EW[edge_index]="<<EW[edge_index]<<std::endl;
//        std::cout<<"done_edges[edge_index]="<<done_edges[edge_index]<<std::endl<<std::endl;


        neigh = indices[edge_index];
        if (neigh<NUMSITES)
        {
            terminal=neigh;
            // update edge outgoing from terminal going to BP_idx_glob
            for (edge_index2=indptr[terminal];edge_index2<indptr[terminal+1];edge_index2++)
            {
                if (indices[edge_index2]==BP_idx_glob)
                {
                    EW[edge_index2]=fabs(d[BP_idx_glob]);
                    done_edges[edge_index2]=1;
                    break;
                }
            }
            // reduce degree of terminal
            degrees_terminals[terminal]-=1;
            // update demand
            d[terminal]+=d[BP_idx_glob];

            //TODO: Remove this
//            std::cout<<"update edge outgoing from terminal going to BP_idx_glob"<<std::endl;
//            std::cout<<"terminal="<<terminal<<std::endl;
//            std::cout<<"edge_index2="<<edge_index2<<std::endl;
//            std::cout<<"indices[edge_index2]="<<indices[edge_index2]<<std::endl;
//            std::cout<<"EW[edge_index2]="<<EW[edge_index2]<<std::endl;
//            std::cout<<"done_edges[edge_index2]="<<done_edges[edge_index2]<<std::endl;
//            std::cout<<"d[terminal]="<<d[terminal]<<std::endl<<std::endl<<std::endl;

            if (degrees_terminals[terminal]==1)
            {
                // update edge outgoing from terminal which has not been updated yet
                for (edge_index2=indptr[terminal];edge_index2<indptr[terminal+1];edge_index2++)
                {
//                    std::cout<<"terminal="<<terminal<<" indptr[terminal]="<<indptr[terminal]<<" indptr[terminal+1]="<<indptr[terminal+1]<<" edge_index2="<<edge_index2<<std::endl;

                    if(done_edges[edge_index2]==0)
                    {
                        EW[edge_index2]=fabs(d[terminal]);
                        done_edges[edge_index2]=1;
                        break;
                    }
                }

                //TODO: Remove this
//                std::cout<<"update edge outgoing from terminal which has not been updated yet"<<std::endl;
//                std::cout<<"BP_idx2="<<indices[edge_index2]<<std::endl;
//                std::cout<<"edge_index2="<<edge_index2<<std::endl;
//                std::cout<<"EW[edge_index2]="<<EW[edge_index2]<<std::endl;
//                std::cout<<"done_edges[edge_index2]="<<done_edges[edge_index2]<<std::endl<<std::endl;

                // update opposite direction edge, i.e. from indices[edge_index2] to terminal
                BP_idx2=indices[edge_index2]-NUMSITES;
                BP_idx2_glob=indices[edge_index2];
                for (edge_index2=indptr[BP_idx2_glob];edge_index2<indptr[BP_idx2_glob+1];edge_index2++)
                {
                    if(indices[edge_index2]==terminal)
                    {
                        EW[edge_index2]=fabs(d[terminal]);
                        done_edges[edge_index2]=1;
                        break;
                    }
                }

                val[BP_idx2]--;
                d[BP_idx2_glob]+=d[terminal];

                //TODO: Remove this
//                std::cout<<"update opposite direction edge, i.e. from indices[edge_index2] to terminal"<<std::endl;
//                std::cout<<"BP_idx2="<<BP_idx2_glob<<std::endl;
//                std::cout<<"edge_index2="<<edge_index2<<std::endl;
//                std::cout<<"indices[edge_index2]="<<indices[edge_index2]<<std::endl;
//                std::cout<<"EW[edge_index2]="<<EW[edge_index2]<<std::endl<<std::endl;
//                std::cout<<"d[BP_idx2_glob]="<<d[BP_idx2_glob]<<std::endl<<std::endl;
//                std::cout<<"----------------------------------------------"<<std::endl;
                // check if neighbor has become leaf
                if(val[BP_idx2] == 1)
                {
                    leafQ[lqp] = BP_idx2;
                    lqp ++;
                }
            }
            continue;
        }
        BP_idx2 = neigh-NUMSITES;
        BP_idx2_glob=neigh;
        for (edge_index=indptr[BP_idx2_glob];edge_index<indptr[BP_idx2_glob+1];edge_index++)
        {
            if(indices[edge_index]==BP_idx_glob){break;}
        }
        EW[edge_index] = fabs(d[BP_idx_glob]);
        done_edges[edge_index] = 1;
        d[BP_idx2_glob]+=d[BP_idx_glob];

        //TODO: Remove this
//        std::cout<<"BP_idx2="<<BP_idx2_glob<<std::endl;
//        std::cout<<"edge_index2="<<edge_index2<<std::endl;
//        std::cout<<"indices[edge_index2]="<<indices[edge_index2]<<std::endl;
//        std::cout<<"EW[edge_index2]="<<EW[edge_index2]<<std::endl<<std::endl;
//        std::cout<<"d[BP_idx2_glob]="<<d[BP_idx2_glob]<<std::endl<<std::endl;
//        std::cout<<"----------------------------------------------"<<std::endl;

        val[BP_idx2]--;
        /* check if neighbor has become leaf? */
        if(val[BP_idx2] == 1)
        {
            leafQ[lqp] = BP_idx2;
            lqp ++;
        }
    }
}

void init_XX(int DIMENSION, int NUMSITES, int NUM_BPS,int NUM_EDGES, int indices[], int indptr[] ,double EW[],
            double demands[], double XX[], double al, double improv_thres, double cost){
    /*calculates the optimal angles at each BP for the convergence checks later. Also initializes the BP positions (in OT configuration).*/
    int d;
    double dist_thres;
    int BP_idx,BP_idx_glob,edge_index,neigh_idx_glob,neigh_idx;
    int lqp=0,counter_neighbors;
    int leafQ[NUM_BPS],val[NUM_BPS];
    int done_edges[NUM_EDGES];



    //find BPs neigboring only a single BP
    for (BP_idx=0;BP_idx<NUM_BPS;BP_idx++)
    {
        val[BP_idx] = 0;
        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++)
        {
            done_edges[edge_index] = 0;
            neigh_idx_glob=indices[edge_index];
            if(neigh_idx_glob>=NUMSITES)
            {
                val[BP_idx]++;
            }else
            {
                done_edges[edge_index]=1;
            }
        }
        if(val[BP_idx]==1)
        {
            leafQ[lqp]=BP_idx;
            lqp++;
        }
    }
    /// INIT coordinates as average of neighbors + noise
    while(lqp>1)
    {
        lqp--;
        BP_idx = leafQ[lqp]; BP_idx_glob=BP_idx+NUMSITES; val[BP_idx]--;

        counter_neighbors=1;
        for (edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++)
        {
           /*find parent BP*/

            neigh_idx_glob=indices[edge_index];
            //////////////////////////////////////////////////////////////
            //TODO: Remove this
//            std::cout<<"edge_index="<<edge_index<<std::endl;
//            std::cout<<"neigh_idx_glob="<<neigh_idx_glob<<std::endl;
//            std::cout<<"BP_idx_glob="<<BP_idx_glob<<std::endl;
//            //print done_edges
//            for (int i=0; i<NUM_EDGES; i++){
//                if (i==0){
//                    std::cout << "done_edges: ";
//                }
//                std::cout << done_edges[i] << " ";
//            }
///////////////////////////////////////////////////////////////////
            if(done_edges[edge_index]!=0)
            {
                dist_thres = (improv_thres*cost)/(pow(fabs(EW[edge_index]),al)*(NUMSITES-2));
                for(d=0;d<DIMENSION;d++)
                {
                    XX[(BP_idx_glob)*DIMENSION+d]*=(counter_neighbors-1);
                    XX[(BP_idx_glob)*DIMENSION+d]+=(XX[neigh_idx_glob*DIMENSION+d]+0.001*dist_thres*(double)(rand()/RAND_MAX));
                    XX[(BP_idx_glob)*DIMENSION+d]/=counter_neighbors;
                }
                counter_neighbors++;
            }
            else
            {
                done_edges[edge_index] = 1;
                neigh_idx = neigh_idx_glob-NUMSITES;

                for (edge_index=indptr[neigh_idx];edge_index<indptr[neigh_idx+1];edge_index++)
                {
                    if(indices[edge_index]==BP_idx_glob){break;}
                }
                done_edges[edge_index] = 1;
                val[neigh_idx]--;
                /* check if neighbor has become leaf? */
                if(val[neigh_idx] == 1)
                {
                    leafQ[lqp] = neigh_idx;
                    lqp ++;
                }
            }
        }
    }
}


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

//    int d;
//    int k,  b;
//    double dist_thres;
//    int BP_idx,edge_index,neigh1,neigh2;
//    int leafQ[NUM_BPS],val[NUM_BPS];
//
//
//    queue<int> BPQ;
//    queue<int> BPQ2;
//    bool BP_done[NUM_BPS];


//    //find BPs neigboring a terminal and push them into the queue
//    for(BP_idx=0;BP_idx<NUM_BPS;BP_idx++){
//        BP_done[BP_idx]=false;
//        for(edge_index=indptr[BP_idx];edge_index<indptr[BP_idx+1];edge_index++)
//        {
//            if (indices[edge_index]<NUMSITES)
//            {
//                BPQ.push(BP_idx);
//                break;
//            }
//        }
//
//    }
//    // iterate over BPs with terminals as neighbor
//    while(!BPQ.empty())
//    {
//        k=BPQ.front();BPQ.pop();
//
//        BPQ2.push(k);
//        while(!BPQ2.empty())
//        {
//            b = BPQ2.front();BPQ2.pop();
//            BP_done[b] = true;
//
//
//            //init BP positions
//            edge_index=indptr[b];
//            neigh1 = indices[edge_index];
//            dist_thres = (improv_thres*cost)/(pow(fabs(EW[edge_index]),al)*(NUMSITES-2));
//            for(d=0;d<DIMENSION;d++)
//            {
//                XX[(b+NUMSITES)*DIMENSION+d] = XX[neigh1*DIMENSION+d]+0.001*dist_thres*(double)(rand()/RAND_MAX);
//            }
//
//            for(edge_index=indptr[b];edge_index<indptr[b+1];edge_index++)
//            {
//                neigh2 = indices[edge_index];
//                if(neigh2>=NUMSITES)
//                {
//                    if(!BP_done[neigh2-NUMSITES])
//                    {
//                        BPQ2.push(neigh2-NUMSITES);
//                    }
//                }
//            }
//        }
//    }
//}


bool check_convergence(double opt_ang[][2], int DIMENSION, int NUMSITES, int adj[][3], double EW[][3], double XX[],double al, double cost, double improv_thres){
    /*calculates the angles between BPs and terminals, checks for V/L-Branching and compares to angles calculated in opt_ang.*/
    int k1,m,l,breaker,d;
    int k;
    int n0,n1,n2;
    double theta1,theta2,norm1,norm2,scprod,vec1,vec2,dist,t,dist_thres,ang_thres;
    double psi,phi,rho;

    n0=n1=n2=0;

    double tol = 1e-10;
    bool not_conv = false;

    for(k=0;k<NUMSITES-2;k++){
        k1=k+NUMSITES;
        //find source; "source" means the edge that has a different flow direction from the other two, so can also be sink. We map to "2sinks-1source"-case
        breaker=0;
        for(m=0;m<3;m++){
            for(l=0;l<3;l++){
                if(m != l){
                    if(EW[k][m]*EW[k][l] >= 0){n1=m;n2=l;breaker=1;break;}
                }
            }
            if(breaker==1){break;}
        }

        for(m=0;m<3;m++){if(m != n1 && m != n2){n0=m;break;}}
        n0 = adj[k][n0];
        n1 = adj[k][n1];
        n2 = adj[k][n2];


        norm1=norm2=scprod=0;
        for(d=0;d<DIMENSION;d++){
            vec1 = XX[n1*DIMENSION+d]-XX[k1*DIMENSION+d];
            vec2 = XX[k1*DIMENSION+d]-XX[n0*DIMENSION+d];
            norm1 += vec1*vec1;
            norm2 += vec2*vec2;
            scprod += vec1*vec2;
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        if(norm1<tol || norm2<tol){continue;}//ignore cases where angles cannot be calculated. These are hopefully already optimal after initial optimization
        theta1 = acos(scprod/(norm1*norm2));

        norm1=norm2=scprod=0;
        for(d=0;d<DIMENSION;d++){
            vec1 = XX[n2*DIMENSION+d]-XX[k1*DIMENSION+d];
            vec2 = XX[k1*DIMENSION+d]-XX[n0*DIMENSION+d];
            norm1 += vec1*vec1;
            norm2 += vec2*vec2;
            scprod += vec1*vec2;
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        if(norm1<tol || norm2<tol){continue;}//ignore cases where angles cannot be calculated. These are hopefully already optimal after initial optimization
        theta2 = acos(scprod/(norm1*norm2));

        dist_thres = (improv_thres*cost)/(pow(fabs(EW[k][m]),al)*(NUMSITES-2));
        ang_thres = dist_thres*cos(theta1+theta2);

        //check for V-Branching
        norm1=norm2=scprod=0;
        for(d=0;d<DIMENSION;d++){
            vec1 = XX[n1*DIMENSION+d]-XX[n0*DIMENSION+d];
            vec2 = XX[n2*DIMENSION+d]-XX[n0*DIMENSION+d];
            norm1 += vec1*vec1;
            norm2 += vec2*vec2;
            scprod += vec1*vec2;
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        psi = acos(scprod/(norm1*norm2));
        if(psi >= opt_ang[k][0]+opt_ang[k][1]){
            dist=0;
            for(d=0;d<DIMENSION;d++){t=XX[k1*DIMENSION+d]-XX[n0*DIMENSION+d];t=t*t;dist+=t;}
            dist = sqrt(dist);
            if(dist>dist_thres){
                for(d=0;d<DIMENSION;d++){XX[k1*DIMENSION+d]=XX[n0*DIMENSION+d]+0.001*dist_thres*(double)(rand()/RAND_MAX);}
                //cout << "V error" << endl;
                not_conv = true;
                continue;
            }
        }

        //check for L1-Branching
        norm1=norm2=scprod=0;
        for(d=0;d<DIMENSION;d++){
            vec1 = XX[n0*DIMENSION+d]-XX[n1*DIMENSION+d];
            vec2 = XX[n2*DIMENSION+d]-XX[n1*DIMENSION+d];
            norm1 += vec1*vec1;
            norm2 += vec2*vec2;
            scprod += vec1*vec2;
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        phi = acos(scprod/(norm1*norm2));
        if(phi >= M_PI-opt_ang[k][1]){
            dist=0;
            for(d=0;d<DIMENSION;d++){t=XX[k1*DIMENSION+d]-XX[n1*DIMENSION+d];t=t*t;dist+=t;}
            dist = sqrt(dist);
            if(dist>dist_thres){
                for(d=0;d<DIMENSION;d++){XX[k1*DIMENSION+d]=XX[n1*DIMENSION+d]+0.001*dist_thres*(double)(rand()/RAND_MAX);}
                //cout << "L1 error" << endl;
                not_conv = true;
                continue;
            }
        }

        //check for L2-Branching
        norm1=norm2=scprod=0;
        for(d=0;d<DIMENSION;d++){
            vec1 = XX[n0*DIMENSION+d]-XX[n2*DIMENSION+d];
            vec2 = XX[n1*DIMENSION+d]-XX[n2*DIMENSION+d];
            norm1 += vec1*vec1;
            norm2 += vec2*vec2;
            scprod += vec1*vec2;
        }
        norm1 = sqrt(norm1);
        norm2 = sqrt(norm2);
        rho = acos(scprod/(norm1*norm2));
        if(rho >= M_PI-opt_ang[k][0]){
            dist=0;
            for(d=0;d<DIMENSION;d++){t=XX[k1*DIMENSION+d]-XX[n2*DIMENSION+d];t=t*t;dist+=t;}
            dist = sqrt(dist);
            if(dist>dist_thres){
                for(d=0;d<DIMENSION;d++){XX[k1*DIMENSION+d]=XX[n2*DIMENSION+d]+0.001*dist_thres*(double)(rand()/RAND_MAX);}
                //cout << "L2 error" << endl;
                not_conv = true;
                continue;
            }
        }

        //calculate angle err
        if(fabs(opt_ang[k][0]-theta1)+fabs(opt_ang[k][1]-theta2) > 2*ang_thres){
            //cout << "angle error: " << opt_ang[k][0] << " " << opt_ang[k][1] << " " << theta1 << " " << theta2 << endl;
            not_conv=true;
        }
    }
    return not_conv;
}




extern "C"
double iterations(int *iter, int DIMENSION, int NUMSITES, int indices[], int indptr[], double EW[], double demands[], double XX[],
                    double al, int NUM_BPS,double improv_thres = 1e-7,bool EW_given=false, double beta=1,
                    bool use_init=true){
    /*iteratively optimizes the BP configuration until improvement threshold is reached*/
//    int count,c,i;
    double cost,cost_old,improv;
//    bool not_conv;
    int NUM_EDGES=2*(NUM_BPS-1)+NUMSITES;
    int max_num_neighs=0;
    int BP_idx;

    double EL[NUM_EDGES];
//    double opt_ang[NUMSITES][2];

    if (EW_given==false){
        calculate_EW_BCST(NUMSITES,NUM_BPS,NUM_EDGES,indices, indptr, EW,demands,al);
    }

    /* determine maximum number neighbors among all BPs */
    for (BP_idx=0;BP_idx<NUM_BPS;BP_idx++)
    {
        if (indptr[BP_idx+1]-indptr[BP_idx]>max_num_neighs)
        {
            max_num_neighs=indptr[BP_idx+1]-indptr[BP_idx];
        }
    }

    cost_old = length(DIMENSION, NUMSITES,NUM_BPS,indices, indptr, EW, EL, XX, al);
    ///TODO: Remove this
    ////////////////////////////////////////////////

//    std::cout << "cost_old: " << cost_old << std::endl;
    ////////////////////////////////////////////////


//    calculate_opt_angle(opt_ang, DIMENSION, NUMSITES, adj, EW, demands, XX, al, improv_thres, cost_old); //calculates optimal angles from flows AND initializes BP positions
    if (use_init)
    {
        init_XX(DIMENSION, NUMSITES, NUM_BPS, NUM_EDGES, indices, indptr, EW, demands, XX, al, improv_thres, cost_old);
    }
    ///TODO: Remove this
    /////////////////////////
//    std::cout << "init_XX done" << std::endl;
//    /*print XX*/
//    for(int i=0;i<NUMSITES+NUM_BPS;i++){
//        for(int j=0;j<DIMENSION;j++){
//            std::cout << XX[i*DIMENSION+j] << " ";
//        }
//        std::cout << std::endl;
//    }
    ////////////////////////////////////////////////


    improv = 1.0;
    *iter = 0;
    do{
        (*iter)++;
        optimize(DIMENSION, NUMSITES,NUM_BPS, NUM_EDGES,max_num_neighs,indices, indptr, EW, EL, XX, al);
        ///TODO: Remove this
        /////////////////////////
        /*print XX*/
//        for(int i=0;i<NUMSITES+NUM_BPS;i++){
//            for(int j=0;j<DIMENSION;j++){
//                std::cout << XX[i*DIMENSION+j] << " ";
//            }
//            std::cout << std::endl;
//        }
        /////////////////////////

        cost=length(DIMENSION, NUMSITES,NUM_BPS,indices, indptr, EW, EL, XX, al);
        improv = (cost_old - cost)/cost_old;
        cost_old =  cost;
//        std::cout << "iter: " << *iter << " cost: " << cost << " improv: " << improv << std::endl;
    }while(improv>improv_thres);
//    i = *iter;
//    for(c=0;c<4;c++){
//        not_conv = check_convergence(opt_ang, DIMENSION, NUMSITES, adj, EW, XX, al, cost, improv_thres);
//        if(not_conv){
//            for(count=0;count<i/5;count++){
//                (*iter)++;
//                optimize(DIMENSION, NUMSITES, adj, EW, EL, XX, al);
//                cost=length(DIMENSION, NUMSITES, adj, EW, EL, XX, al);
//            }
//        }
//        else{break;}
//    }
    return cost;

}

