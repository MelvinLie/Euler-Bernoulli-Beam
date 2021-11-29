import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


class EulerBernoulliBeamElement:

    def __init__(self,pos1,pos2,dof_table,E,I,m,color='C0',linewidth=3,label="None"):
        self.pos1 = pos1
        self.pos2 = pos2
        self.global_dofs = dof_table
        self.e = E
        self.i = I
        self.m = m
        self.l = np.sqrt((pos2[0]-pos1[0])**2
                            + (pos2[1]-pos1[1])**2
                            + (pos2[2]-pos1[2])**2)

        self.color = color
        self.linewidth = linewidth
        self.label = label

        self.G = 9.80665 #[m/s^2]



    def print_report(self):
        print('--------------------------')
        print('{}'.format(self.label))
        print('Pos 1 = ({} , {} , {}), Pos 2 = ({} , {} , {})'.format(self.pos1[0],
                                                                        self.pos1[1],
                                                                        self.pos1[2],
                                                                        self.pos2[0],
                                                                        self.pos2[1],
                                                                        self.pos2[2]))
        print('DoFs = {}'.format(self.global_dofs))
        print('--------------------------')

    def plot(self,dofs,ax,label='FEM'):
        #simple linear interpolation
        if(label == 'None'):
            ax.plot([self.pos1[2],self.pos2[2]],[dofs[self.global_dofs[0]],dofs[self.global_dofs[2]]],color=self.color,linewidth=self.linewidth)
        else:
            ax.plot([self.pos1[2],self.pos2[2]],[dofs[self.global_dofs[0]],dofs[self.global_dofs[2]]],color=self.color,linewidth=self.linewidth,label=label)

    def compute_M(self):

        M = self.m*self.l/420.*np.array([[  156       ,  22*self.l    ,  54         ,   -13*self.l   ],
                                         [ 22*self.l  ,   4*self.l**2 ,  13*self.l  ,  -3*self.l**2  ],
                                         [  54        ,  13*self.l    ,  156        , -22*self.l     ],
                                         [ -13*self.l , -3*self.l**2  , -22*self.l  ,   4*self.l**2  ]])

        return M


    def compute_K(self):

        K = self.e*self.i/self.l**3*np.array([[ 12       ,  6*self.l    , -12       ,  6*self.l    ],
                                              [ 6*self.l ,  4*self.l**2 , -6*self.l ,  2*self.l**2 ],
                                              [ -12      , -6*self.l    ,  12       , -6*self.l    ],
                                              [ 6*self.l ,  2*self.l**2 , -6*self.l ,  4*self.l**2 ]])
        return K


    def compute_p(self):


        return -1.*self.m*self.G*self.l*np.array([ 0.5 , self.l/12 , 0.5 , -self.l/12 ])


    def evaluate(self,z,dofs):
        ret_val = 0.

        z_hat = z - self.pos1[2]

        ret_val += (2*z_hat**3       - 3*self.l*z_hat**2    + self.l**3      )*dofs[self.global_dofs[0]]
        ret_val += (self.l*z_hat**3  - 2*self.l**2*z_hat**2 + self.l**3*z_hat)*dofs[self.global_dofs[1]]
        ret_val += (-2*z_hat**3      + 3*self.l*z_hat**2                     )*dofs[self.global_dofs[2]]
        ret_val += (self.l*z_hat**3  - self.l**2*z_hat**2                    )*dofs[self.global_dofs[3]]

        ret_val /= self.l**3

        return ret_val

    def evaluate_slope(self,z,dofs):
        ret_val = 0.

        z_hat = z - self.pos1[2]

        ret_val += (6.*z_hat**2       - 6.*self.l*z_hat                  )*dofs[self.global_dofs[0]]
        ret_val += (3.*self.l*z_hat**2  - 4.*self.l**2*z_hat + self.l**3 )*dofs[self.global_dofs[1]]
        ret_val += (-6.*z_hat**2      + 6.*self.l*z_hat                  )*dofs[self.global_dofs[2]]
        ret_val += (3.*self.l*z_hat**2  - 2.*self.l**2*z_hat             )*dofs[self.global_dofs[3]]

        ret_val /= self.l**3

        return ret_val

    def add_mass(self,m):

        self.m += m/self.l

    def evaluate_basis_fcns(self,z):

        z_hat = z - self.pos1[2]

        return np.array([2*z_hat**3       - 3*self.l*z_hat**2    + self.l**3,
                            self.l*z_hat**3  - 2*self.l**2*z_hat**2 + self.l**3*z_hat,
                            -2*z_hat**3      + 3*self.l*z_hat**2,
                            self.l*z_hat**3  - self.l**2*z_hat**2])/self.l**3




class EulerBernoulliBeamAssembly:

    def __init__(self):
        self.knots = np.array([])
        self.global_dof_indices = []
        self.num_dofs = 0
        self.beam_elements = []
        self.stiness_bc = []


    def add_knot(self,knot):
        if self.knots.shape[0] == 0:
            self.knots.resize((1,3))
            self.knots[0,:] = knot

        else:

            self.knots = np.append(self.knots,np.array([knot]),axis=0)

        self.global_dof_indices.append([self.num_dofs,self.num_dofs+1])

    def connect_knots(self,knot1,knot2,num_elements,E,I,m,color='C0',linewidth=3,label="None"):

        indx_1 = -1
        indx_2 = -1

        for i,k in enumerate(self.knots):
            if( ( k[0] == knot1[0] ) and ( k[1] == knot1[1] ) and ( k[2] == knot1[2] ) ):
                indx_1 = i

            if( ( k[0] == knot2[0] ) and ( k[1] == knot2[1] ) and ( k[2] == knot2[2] ) ):
                indx_2 = i

            if ((indx_1 != -1) and (indx_2 != -1)):
                break

        if indx_1 == -1:
            self.add_knot(knot1)
            dofs_1 = [self.num_dofs,self.num_dofs+1]
            self.num_dofs += 2
        else:
            dofs_1 = self.global_dof_indices[indx_1]

        if indx_2 == -1:
            self.add_knot(knot2)
            dofs_2 = [self.num_dofs,self.num_dofs+1]
            self.num_dofs += 2
        else:
            dofs_2 = self.global_dof_indices[indx_2]

        dr = (knot2-knot1)/num_elements

        if(num_elements > 1):

            #append first element
            self.beam_elements.append(EulerBernoulliBeamElement(knot1,knot1+dr,[dofs_1[0],dofs_1[1],self.num_dofs,self.num_dofs+1],E,I,m,color,linewidth,label))
            self.num_dofs += 2

            for i in range(1,num_elements-1):
                self.beam_elements.append(EulerBernoulliBeamElement(knot1+i*dr,knot1+(i+1)*dr,[self.num_dofs-2,self.num_dofs-1,self.num_dofs,self.num_dofs+1],E,I,m,color,linewidth,label))
                self.num_dofs += 2

            #append last element
            self.beam_elements.append(EulerBernoulliBeamElement(knot2-dr,knot2,[self.num_dofs-2,self.num_dofs-1,dofs_2[0],dofs_2[1]],E,I,m,color,linewidth,label))

        else:
            #append element between knots
            self.beam_elements.append(EulerBernoulliBeamElement(knot1,knot1+dr,[dofs_1[0],dofs_1[1],dofs_2[0],dofs_2[1]],E,I,m,color,linewidth,label))

    def connect_knots_segmented(self,knot1,knot2,num_elements,E_list,I_list,m_list,color='C0',linewidth=3,label="None"):

        indx_1 = -1
        indx_2 = -1

        for i,k in enumerate(self.knots):
            if( ( k[0] == knot1[0] ) and ( k[1] == knot1[1] ) and ( k[2] == knot1[2] ) ):
                indx_1 = i

            if( ( k[0] == knot2[0] ) and ( k[1] == knot2[1] ) and ( k[2] == knot2[2] ) ):
                indx_2 = i

            if ((indx_1 != -1) and (indx_2 != -1)):
                break

        if indx_1 == -1:
            self.add_knot(knot1)
            dofs_1 = [self.num_dofs,self.num_dofs+1]
            self.num_dofs += 2
        else:
            dofs_1 = self.global_dof_indices[indx_1]

        if indx_2 == -1:
            self.add_knot(knot2)
            dofs_2 = [self.num_dofs,self.num_dofs+1]
            self.num_dofs += 2
        else:
            dofs_2 = self.global_dof_indices[indx_2]

        dr = (knot2-knot1)/num_elements

        if(num_elements > 1):

            #append first element
            for j in range(len(E_list)):
                self.beam_elements.append(EulerBernoulliBeamElement(knot1,knot1+dr,[dofs_1[0],dofs_1[1],self.num_dofs,self.num_dofs+1],E_list[j],I_list[j],m_list[j],color,linewidth,label))
            self.num_dofs += 2

            for i in range(1,num_elements-1):
                for j in range(len(E_list)):
                    self.beam_elements.append(EulerBernoulliBeamElement(knot1+i*dr,knot1+(i+1)*dr,[self.num_dofs-2,self.num_dofs-1,self.num_dofs,self.num_dofs+1],E_list[j],I_list[j],m_list[j],color,linewidth,label))
                self.num_dofs += 2

            #append last element
            for j in range(len(E_list)):
                self.beam_elements.append(EulerBernoulliBeamElement(knot2-dr,knot2,[self.num_dofs-2,self.num_dofs-1,dofs_2[0],dofs_2[1]],E_list[j],I_list[j],m_list[j],color,linewidth,label))

        else:
            #append element between knots
            for j in range(len(E_list)):
                self.beam_elements.append(EulerBernoulliBeamElement(knot1,knot1+dr,[dofs_1[0],dofs_1[1],dofs_2[0],dofs_2[1]],E_list[j],I_list[j],m_list[j],color,linewidth,label))

    def connect_knots_inhomogeneous(self,knot1,knot2,num_elements,beam_params,color='C0',linewidth=3,label="None"):

        indx_1 = -1
        indx_2 = -1

        for i,k in enumerate(self.knots):
            if( ( k[0] == knot1[0] ) and ( k[1] == knot1[1] ) and ( k[2] == knot1[2] ) ):
                indx_1 = i

            if( ( k[0] == knot2[0] ) and ( k[1] == knot2[1] ) and ( k[2] == knot2[2] ) ):
                indx_2 = i

            if ((indx_1 != -1) and (indx_2 != -1)):
                break

        if indx_1 == -1:
            self.add_knot(knot1)
            dofs_1 = [self.num_dofs,self.num_dofs+1]
            self.num_dofs += 2
        else:
            dofs_1 = self.global_dof_indices[indx_1]

        if indx_2 == -1:
            self.add_knot(knot2)
            dofs_2 = [self.num_dofs,self.num_dofs+1]
            self.num_dofs += 2
        else:
            dofs_2 = self.global_dof_indices[indx_2]

        dr = (knot2-knot1)/num_elements

        if(num_elements > 1):

            E,I,m = beam_params(knot1+0.5*dr)

            #append first element
            self.beam_elements.append(EulerBernoulliBeamElement(knot1,knot1+dr,[dofs_1[0],dofs_1[1],self.num_dofs,self.num_dofs+1],E,I,m,color,linewidth,label))
            self.num_dofs += 2

            for i in range(1,num_elements-1):

                E,I,m = beam_params(knot1+(i+0.5)*dr)

                self.beam_elements.append(EulerBernoulliBeamElement(knot1+i*dr,knot1+(i+1)*dr,[self.num_dofs-2,self.num_dofs-1,self.num_dofs,self.num_dofs+1],E,I,m,color,linewidth,label))
                self.num_dofs += 2

            #append last element
            E,I,m = beam_params(knot2-0.5*dr)

            self.beam_elements.append(EulerBernoulliBeamElement(knot2-dr,knot2,[self.num_dofs-2,self.num_dofs-1,dofs_2[0],dofs_2[1]],E,I,m,color,linewidth,label))

        else:
            #append element between knots
            E,I,m = beam_params(knot1+0.5*dr)

            self.beam_elements.append(EulerBernoulliBeamElement(knot1,knot1+dr,[dofs_1[0],dofs_1[1],dofs_2[0],dofs_2[1]],E,I,m,color,linewidth,label))

    def print_report(self):
        for be in self.beam_elements:
            be.print_report()

    def assemble_M(self):

        rows = []
        cols = []
        data = []

        for i,be in enumerate(self.beam_elements):

            m = be.compute_M()

            for j in range(4):
                for l in range(4):

                    rows.append(be.global_dofs[j])
                    cols.append(be.global_dofs[l])
                    data.append(m[j,l])


        M = csr_matrix((data,(rows,cols)),shape=(self.num_dofs,self.num_dofs))

        return M

    def assemble_K(self):

        rows = []
        cols = []
        data = []

        for i,be in enumerate(self.beam_elements):

            k = be.compute_K()

            for j in range(4):
                for l in range(4):

                    rows.append(be.global_dofs[j])
                    cols.append(be.global_dofs[l])
                    data.append(k[j,l])




        for i,stiffness_bc in enumerate(self.stiness_bc):

            for e,be in enumerate(self.beam_elements):
                if (be.pos2[2] >= stiffness_bc[0]) and (be.pos1[2] <= stiffness_bc[0]):


                    base_fcns = be.evaluate_basis_fcns(stiffness_bc[0])
                    k = stiffness_bc[1]*np.outer(base_fcns,base_fcns)

                    for j in range(4):
                        for l in range(4):
                            rows.append(be.global_dofs[j])
                            cols.append(be.global_dofs[l])

                            data.append(k[j,l])

        K = csr_matrix((data,(rows,cols)),shape=(self.num_dofs,self.num_dofs))


        return K

    def assemble_p(self):

        p = np.zeros((self.num_dofs,))

        for i,be in enumerate(self.beam_elements):

            pp = be.compute_p()

            p[be.global_dofs[0]] += pp[0]
            p[be.global_dofs[1]] += pp[1]
            p[be.global_dofs[2]] += pp[2]
            p[be.global_dofs[3]] += pp[3]

        return p

    def plot_deformation(self,dofs,ax,label='FEM'):
        #fig,ax = plt.subplots()

        for i,be in enumerate(self.beam_elements):
            if(i == 0):
                be.plot(dofs,ax,label=label)
            else:
                be.plot(dofs,ax,label='None')

        #plt.show()

    def evaluate_slope(self,dofs,z):

        el_list = self.get_local_elements(z)

        return el_list[0].evaluate_slope(z,dofs)


    def compute_displacement(self,z_array,dofs,label="None"):

        if(np.iscomplexobj(dofs)):
            w = 1j*np.zeros(len(z_array))
        else:
            w = np.zeros(len(z_array))

        for i,zz in enumerate(z_array):

            el_list = self.get_local_elements(zz)
            num_el = 0

            for be in el_list:

                if be.label == label:
                    w[i] += be.evaluate(zz,dofs)
                    num_el += 1
            #if(num_el == 0):
            #    #print(zz)
            #    #print(el_list[0].print_report())
            w[i] /= num_el

        return w

    def get_local_elements(self,z):

        el_list = []

        for i,be in enumerate(self.beam_elements):
            if (be.pos2[2] >= z) and (be.pos1[2] <= z):
                el_list.append(be)

        return el_list

    def get_knot_dofs(self,knot):

        dofs = []

        for i,k in enumerate(self.knots):
            if( ( k[0] == knot[0] ) and ( k[1] == knot[1] ) and ( k[2] == knot[2] ) ):
                dofs.append(self.global_dof_indices[i])

        return dofs

    def add_mass(self,z,m):

        el_list = self.get_local_elements(z)

        #print('Add mass at z = {}'.format(z))


        for be in el_list:
            print(be.print_report())
            be.add_mass(m)

    def add_mass_between_positions(self,z_0,z_1,m):

        for i,be in enumerate(self.beam_elements):
            z_ctr = 0.5*(be.pos2[2]+be.pos1[2])
            if (z_ctr < z_1) and (z_ctr > z_0):
                be.add_mass(m)

    def add_distributed_mass_between_positions(self,z_0,z_1,m):

        for i,be in enumerate(self.beam_elements):
            z_ctr = 0.5*(be.pos2[2]+be.pos1[2])
            if (z_ctr < z_1) and (z_ctr > z_0):
                len_el = abs(be.pos2[2]-be.pos1[2])
                be.add_mass(m*len_el)

    def apply_stiffness(self,z,k):

        self.stiness_bc.append(np.array([z,k]))

        #for i,be in enumerate(self.beam_elements):
        #    if (be.pos2[2] >= z) and (be.pos1[2] <= z):
        #        el_list.append(be)

        #return el_list

def compute_second_moment_of_area(x,y):

    Ix = 0
    Iy = 0


    #int y**2 dx dy
    for i,xx in enumerate(x[:-1]):
        Ix += (x[i]*y[i+1] - x[i+1]*y[i]) * (y[i]**2 + y[i]*y[i+1] + y[i+1]**2)

    #int x**2 dx dy
    for i,xx in enumerate(x[:-1]):
        Iy += (x[i]*y[i+1] - x[i+1]*y[i]) * (x[i]**2 + x[i]*x[i+1] + x[i+1]**2)



    return Ix/12.,Iy/12.

def compute_area(x,y):

    A = 0

    for i,xx in enumerate(x[:-1]):
        A += (x[i+1]+x[i])*(y[i+1]-y[i])

    return A/2.

def estimate_length_given_frequency(f,E,I,m):
    L = np.sqrt(3.516/2/np.pi/f)*(E*I/m)**0.25

    return L

if __name__ == "__main__":

    '''
    p = np.zeros((13,2))
    p[0,:] = np.array([0.02,0.])
    p[1,:] = np.array([0.03,0.])
    p[2,:] = np.array([0.03,0.025])
    p[3,:] = np.array([0.02,0.038])
    p[4,:] = np.array([0.01,0.042])
    p[5,:] = np.array([-0.01,0.042])
    p[6,:] = np.array([-0.02,0.038])
    p[7,:] = np.array([-0.03,0.025])
    p[8,:] = np.array([-0.03,0.])
    p[9,:] = np.array([-0.02,0.])
    p[10,:] = np.array([-0.02,0.025])
    p[11,:] = np.array([0.,0.038])
    p[12,:] = np.array([0.02,0.025])
    p[:,1] -= 0.015

    Ix,Iy = compute_second_moment_of_area(p[:,0],p[:,1])


    print('Ix brass part = {}'.format(Ix))
    print('Iy brass part = {}'.format(Iy))

    A = compute_area(p[:,0],p[:,1])

    print('Area brass part = {}'.format(A))
    print('Volume brass part = {}'.format(A*0.03))
    print('Mass brass part = {}'.format(A*0.03*8730))


    p2 = np.zeros((7,2))
    p2[0,:] = np.array([0.02,0.])
    p2[1,:] = np.array([0.03,0.])
    p2[2,:] = np.array([0.03,-0.015])
    p2[3,:] = np.array([-0.03,-0.015])
    p2[4,:] = np.array([-0.03,0.])
    p2[5,:] = np.array([-0.02,0.])
    p2[6,:] = np.array([0.0,-0.0075])
    p2[:,1] -= 0.015

    Ix,Iy = compute_second_moment_of_area(p2[:,0],p2[:,1])

    print('Ix alu counter part = {}'.format(Ix))
    print('Iy alu counter  part = {}'.format(Iy))

    A = compute_area(p2[:,0],p2[:,1])

    print('Area alu counter part = {}'.format(A))


    p3 = np.zeros((6,2))
    p3[0,:] = p2[3,:] + np.array([0.0,0.004])
    p3[1,:] = p3[0,:] + np.array([0.005,-0.004])
    p3[2,:] = p3[1,:] + np.array([0.015,0.])
    p3[3,:] = p3[2,:] + np.array([0.0,0.004])
    p3[4,:] = p3[3,:] + np.array([-0.005,0.004])
    p3[5,:] = p3[4,:] + np.array([-0.015,0.0])

    Ix,Iy = compute_second_moment_of_area(p3[:,0],p3[:,1])

    print('Ix alu ground plate left = {}'.format(Ix))
    print('Iy alu ground plate left = {}'.format(Iy))

    A = compute_area(p3[:,0],p3[:,1])

    print('Area alu ground plate left = {}'.format(A))

    p4 = p3.copy()
    p4[:,0] *= -1

    Ix,Iy = compute_second_moment_of_area(p4[:,0],p4[:,1])

    A = compute_area(p4[:,0],p4[:,1])

    print('Ix alu ground plate right = {}'.format(Ix))
    print('Iy alu ground plate right = {}'.format(Iy))

    print('Area alu ground plate right = {}'.format(A))

    p5 = np.zeros((6,2))
    p5[0,:] = p3[5,:]
    p5[1,:] = p3[0,:]
    p5[2,:] = p3[1,:]
    p5[3,:] = p4[1,:]
    p5[4,:] = p4[0,:]
    p5[5,:] = p4[5,:]


    Ix,Iy = compute_second_moment_of_area(p5[:,0],p5[:,1])

    A = compute_area(p5[:,0],p5[:,1])

    print('***************************************')
    print('Segment 1:')
    print('p5 A = {}'.format(A))

    print('Ix alu ground plate big = {}'.format(Ix))
    print('Iy alu ground plate big = {}'.format(Iy))

    print('Area alu ground plate big = {}'.format(A))
    print('***************************************')

    '''

    print('***************************************')
    print('Test Parallel axis theorem')
    print('***************************************')
    p_test = np.array([[-1.,-1.],
                        [1.,-1.],
                        [1.,1.],
                        [-1.,1.],
                        [-1.,-1.]])


    Ix,Iy = compute_second_moment_of_area(p_test[:,0],p_test[:,1])

    d = 3.
    p_test[:,1] += d

    Ix_2,Iy_2 = compute_second_moment_of_area(p_test[:,0],p_test[:,1])


    A = compute_area(p_test[:,0],p_test[:,1])

    print('Parallel axis theorem:')
    print('Ix = {}'.format(Ix+A*d**2))
    print('Direct integration:')
    print('Ix = {}'.format(Ix_2))

    print('***************************************')
    print('Brass clamp')
    print('***************************************')
    p_brass = np.zeros((14,2))
    p_brass[0,:] = np.array([0.02,0.])
    p_brass[1,:] = np.array([0.03,0.])
    p_brass[2,:] = np.array([0.03,0.025])
    p_brass[3,:] = np.array([0.02,0.038])
    p_brass[4,:] = np.array([0.01,0.042])
    p_brass[5,:] = np.array([-0.01,0.042])
    p_brass[6,:] = np.array([-0.02,0.038])
    p_brass[7,:] = np.array([-0.03,0.025])
    p_brass[8,:] = np.array([-0.03,0.])
    p_brass[9,:] = np.array([-0.02,0.])
    p_brass[10,:] = np.array([-0.02,0.025])
    p_brass[11,:] = np.array([0.,0.038])
    p_brass[12,:] = np.array([0.02,0.025])
    p_brass[13,:] = np.array([0.02,0.])

    p_brass[:,1] -= 0.015

    Ix,Iy = compute_second_moment_of_area(p_brass[:,0],p_brass[:,1])


    print('Ix brass part = {}'.format(Ix))
    print('Iy brass part = {}'.format(Iy))

    A = compute_area(p_brass[:,0],p_brass[:,1])

    print('Area = {}'.format(A))


    vertical_shift = -0.026

    print('***************************************')
    print('Arm holder')
    print('***************************************')
    print('Segment 1:')
    p_s1 = np.array([[0.0255,-0.004],
                  [0.03,0.0005],
                  [0.03,0.011],
                  [0.0165,0.011],
                  [0.006,0.0065],
                  [0.006,0.004],
                  [-0.006,0.004],
                  [-0.006,0.0065],
                  [-0.0165,0.011],
                  [-0.03,0.011],
                  [-0.03,0.0005],
                  [-0.0255,-0.004],
                  [0.0255,-0.004]])

    p_s1[:,1] += vertical_shift

    Ix,Iy = compute_second_moment_of_area(p_s1[:,0],p_s1[:,1])

    A = compute_area(p_s1[:,0],p_s1[:,1])

    print('Ix = {}'.format(Ix))
    print('Iy = {}'.format(Iy))

    print('Area = {}'.format(A))


    print('***************************************')
    print('Segment 2:')
    p_s2 = np.array([[0.0255,-0.004],
                  [0.03,0.0005],
                  [0.03,0.004],
                  [-0.03,0.004],
                  [-0.03,0.0005],
                  [-0.0255,-0.004],
                  [0.0255,-0.004]])

    p_s2[:,1] += vertical_shift

    Ix,Iy = compute_second_moment_of_area(p_s2[:,0],p_s2[:,1])

    A = compute_area(p_s2[:,0],p_s2[:,1])

    print('Ix = {}'.format(Ix))
    print('Iy = {}'.format(Iy))

    print('Area = {}'.format(A))



    print('***************************************')
    print('Segment 3:')
    p_s3 = np.array([[0.0255,-0.004],
                  [0.03,0.0005],
                  [0.03,0.004],
                  [0.01,0.004],
                  [0.01,-0.004],
                  [0.0255,-0.004]])

    p_s3[:,1] += vertical_shift

    Ix,Iy = compute_second_moment_of_area(p_s3[:,0],p_s3[:,1])

    A = compute_area(p_s3[:,0],p_s3[:,1])

    print('Ix = {}'.format(Ix))
    print('Iy = {}'.format(Iy))

    print('Area = {}'.format(A))


    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.plot(p_s1[:,0],p_s1[:,1],label = 'Segment 1')
    ax.plot(p_brass[:,0],p_brass[:,1],color='C0')
    ax.set_xlim((-0.033,0.033))
    ax.set_ylim((-0.033,0.033))
    #ax.set_ylim((-0.005,0.016))
    ax.legend()

    ax = fig.add_subplot(222)
    ax.plot(p_s2[:,0],p_s2[:,1],label = 'Segment 2')
    ax.set_xlim((-0.033,0.033))
    ax.set_ylim((-0.033,0.033))
    #ax.set_ylim((-0.005,0.016))
    ax.legend()

    ax = fig.add_subplot(223)
    ax.plot(p_s3[:,0],p_s3[:,1],color='C0')
    ax.plot(-1*p_s3[:,0],p_s3[:,1],color='C0',label = 'Segment 3')
    ax.set_xlim((-0.033,0.033))
    ax.set_ylim((-0.033,0.033))
    #ax.set_ylim((-0.005,0.016))
    ax.legend()

    ax = fig.add_subplot(224)
    ax.plot(p_brass[:,0],p_brass[:,1],color='C0')
    ax.set_xlim((-0.033,0.033))
    ax.set_ylim((-0.033,0.033))
    #ax.set_ylim((-0.005,0.016))
    ax.legend()

    plt.tight_layout()
    plt.show()

    '''
    fig,ax = plt.subplots()
    ax.plot(p[:,0],p[:,1],label = 'brass')
    ax.plot(p2[:,0],p2[:,1],label = 'alu counter part')
    ax.plot(p3[:,0],p3[:,1],label = 'alu ground plate left')
    ax.plot(p4[:,0],p4[:,1],label = 'alu ground plate right')
    ax.plot(p5[:,0],p5[:,1],label = 'alu ground plate big')
    ax.legend()
    plt.show()
    '''
    exit()
