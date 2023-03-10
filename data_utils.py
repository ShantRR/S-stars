import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from scipy.integrate import odeint


def coords(ra, dec, dist):
            """Equatorial coordinates to Cartesian coordinates"""
            x = dist*np.cos(dec)*np.cos(ra)
            y = dist*np.cos(dec)*np.sin(ra)
            z = dist*np.sin(dec)

            return x, y, z

def convert_partial_year(number):

    year = int(number)
    d = timedelta(days=(number - year)*365)
    day_one = datetime(year,1,1)
    date = d + day_one
    return date.strftime("%Y %m %d %H %M %S")        

def ellipse(phi,e,p):
    
    r = p/(1+e*np.cos(phi))
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return x,y,r

def SkyPlaneTransform(names, call_path='..',save_path='../Transformed_data/'):
    
    table_S = pd.read_html(call_path + '/Original_data/html.htm')
    table_orbital_params = pd.read_html(call_path + '/Original_data/tab2.htm')
    df_stars = table_S[0]
    df_params = table_orbital_params[0]
    fig = plt.figure(figsize=(23,10))

    data_stars_names = [col[4:-7] for col in df_stars.columns[2:-1:3]]
    data_stars_params_names = df_params['Star (f)---'].tolist()
    names_posible_convert = list(set(data_stars_names) & set(data_stars_params_names))
    
    if names == 'All':
        names = names_posible_convert
        print(f'converting the all stars data --- {names}')
    else:
        names = list(set(names_posible_convert) & set(names))
        print(f'converting the folowing stars data --- {names}')
        
    dataset_dict = {}
    for name in names:

        oRa = 'oRA-'+ name + ' (e)mas'
        oDE = 'oDE-'+ name + ' (e)mas'

        df_star = df_stars[df_stars[oRa].notna()].copy()

        Year = df_star['Dateyr'].to_numpy().astype(np.float64)

        df_star[oRa] = df_star[oRa].str.split().str[0]
        df_star[oDE] = df_star[oDE].str.split().str[0]

        RA = df_star[oRa].to_numpy().astype(np.float64)*10**(-3) # from milli-second of arc [mas] to to arcsec ['']
        DE = df_star[oDE].to_numpy().astype(np.float64)*10**(-3)
        
        
        # partial year to time difference
        
        time_obs = [convert_partial_year(t).split() for t in Year]
        time_obs = np.asarray([list(map(int, i)) for i in time_obs])

        to_sec = np.array([3.154*10**(7), 2.628*10**6, 86400, 3600, 60, 1])

        t_sec = (time_obs-time_obs[0]).dot(to_sec) # time difference between observations, first observetion is starting point
        t_mnth = t_sec * 3.80517*10**(-7)
        t_yr = t_sec * 3.171*10**(-8)


        # real orbit parameters 
        i = df_params.loc[df_params['Star (f)---'] == name, 'i (e)deg'].iloc[0].split()[0]
        i = np.array(i, dtype=float)*np.pi/180

        Omega = df_params.loc[df_params['Star (f)---'] == name, 'Omega (e)deg'].iloc[0].split()[0]
        Omega = np.array(Omega, dtype=float)*np.pi/180

        omega = df_params.loc[df_params['Star (f)---'] == name, 'w (e)deg'].iloc[0].split()[0]
        omega = np.array(omega, dtype=float)*np.pi/180
        
       


        # Thiele-Innes constants 
        A = np.cos(Omega)*np.cos(omega) - np.sin(Omega)*np.sin(omega)*np.cos(i)
        B = np.sin(Omega)*np.cos(omega) + np.cos(Omega)*np.sin(omega)*np.cos(i)
        C = np.sin(omega)*np.sin(i)
        F = -np.cos(Omega)*np.sin(omega) - np.sin(Omega)*np.cos(omega)*np.cos(i)
        G = -np.sin(Omega)*np.sin(omega) + np.cos(Omega)*np.cos(omega)*np.cos(i)
        H = np.cos(omega)*np.sin(i)


        distance = 8200 * 206265 #3.086*10**16  # 8200 [pc]  

        _,X, Y = coords(RA*np.pi/(180*3600) , DE*np.pi/(180*3600), distance)
        x_au = []


        for j in range(len(Y)):
            a = np.array([[B, G], [A, F]])
            b = np.array([X[j], Y[j]])
            x_au.append(np.linalg.solve(a,b).tolist())

        x_au = np.array(x_au)
        min_d = min(np.sqrt(x_au.T[0]**2+x_au.T[1]**2))
        max_d = max(np.sqrt(x_au.T[0]**2+x_au.T[1]**2))
        x, y = x_au.T[0]/max_d, x_au.T[1]/max_d
        
        e_S = df_params.loc[df_params['Star (f)---'] == name, 'e (e)---'].iloc[0].split()[0]
        a_S = df_params.loc[df_params['Star (f)---'] == name, 'a (e)arcsec'].iloc[0].split()[0]
        e_S = np.array(e_S, dtype=float)
        a_S = np.array(a_S, dtype=float)
        
        a_rad = a_S*4.84814e-6     # [arcsec] to [rad]
        a_au = 2*distance*np.tan(a_rad/2)
        p_au = a_au*(1-e_S**2)
        
        phi_S = np.linspace(0, 2*np.pi,200)
        x_S,y_S,_ = ellipse(phi_S,e_S,p_au)
        
        
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(122)
        ax1.scatter(x_au.T[0],x_au.T[1], s=25, marker='+', label=name)
        ax1.plot(x_S,y_S)
        ax2.scatter(DE,RA, s=25, marker='+', label=name)

        ax1.legend()
        ax2.legend()
        ax1.set_title('$(x,y)$ [Au]',fontsize=18)
        ax2.set_title('(RA,De)',fontsize=18)
#         plt.close()

        dataset = pd.DataFrame({'Fractional year': Year,
                                'dt [year]'      : t_yr, 
                                'Right Ascension': RA, 
                                'Declination'    : DE,
                                'Real x [Au]'    : x_au.T[0],
                                'Real y [Au]'    : x_au.T[1],
                                'Normalized x'   : x,
                                'Normalized y'   : y})
        dataset_dict[name] = dataset

        dataset.to_html(save_path+'data'+name+'.html')
        print('Saved '+'\x1b[1;31m'+f'{name} '+'\x1b[0m'+ 'data in the' + '  \033[1m' + f'{save_path}data{name}.html' + '\033[0;0m  ' + 'file')
    plt.show()
    fig.savefig('Transformation.png', dpi=300)
    
    return dataset_dict



def star_data_load(name,call_path='..'):
    table_S = pd.read_html(call_path + "/Transformed_data/data"+name +".html")
    df = table_S[0]
    
    ra = df['Right Ascension'].to_numpy()
    de = df['Declination'].to_numpy()
    
    x_au = df['Real x [Au]'].to_numpy()
    y_au = df['Real y [Au]'].to_numpy()
    
    t = df['dt [year]'].to_numpy()
    
    return x_au, y_au, t, ra, de

def Stars_Info(stars_name,call_path='..',show=True):
    
    distance = 8300 * 206265 # [Au]
    table_params = pd.read_html(call_path + "/Original_data/tab2.htm")
    df_params = table_params[0]
    
    
    
    x_stars = {S: star_data_load(S,call_path)[0] for S in stars_name}
    y_stars = {S: star_data_load(S,call_path)[1] for S in stars_name}
    time_obs = {S: star_data_load(S,call_path)[2] for S in stars_name}
    
    phi_stars = {S: np.mod(np.arctan2(y_stars[S],x_stars[S]),2*np.pi) for S in stars_name}
    r_stars = {S: np.sqrt(x_stars[S]**2 + y_stars[S]**2) for S in stars_name}
    u_stars = {S: 1/r_stars[S] for S in stars_name}
    
    x_stars_min_max = [[int(min(i)), int(max(i))] for i in x_stars.values()]
    y_stars_min_max = [[int(min(i)), int(max(i))] for i in y_stars.values()]
    data_count = [len(x_stars[S]) for S in stars_name]
    time_duration =[int(t[-1]) for t in time_obs.values()]
    
    e = [df_params.loc[df_params['Star (f)---'] == name, 'e (e)---'].iloc[0].split()[0] for name in stars_name]
    a = [df_params.loc[df_params['Star (f)---'] == name, 'a (e)arcsec'].iloc[0].split()[0] for name in stars_name]
    e = np.array(e, dtype=float)
    a = np.array(a, dtype=float)
    
    a_rad = a*4.84814e-6     # [arcsec] to [rad]
    a_au = 2*distance*np.tan(a_rad/2)
    p_au = a_au*(1-e**2)
    
    df_info = pd.DataFrame({
                    'Name'                : stars_name,
                    'Data count'          : data_count,
                    'Time duration [year]': time_duration,
                    'X range [Au]'        : x_stars_min_max,
                    'Y range [Au]'        : y_stars_min_max,
                    'e'                   : e,
                    'a [arcsec]'          : a ,
                    'a [Au]'              : a_au,
                    'p [Au]'              : p_au})
    
    df_info = df_info.set_index('Name')
    df_info_display = df_info.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    df_info_display.set_properties(**{'text-align': 'center'})
    if show:
        display(df_info_display)
    return x_stars, y_stars, r_stars, u_stars, phi_stars, df_info



def EllipseNorm(x,y,e,p, coeff=1):

    ellipse_area = np.pi*p**2/(1-e**2)**(3/2)
    ellipse_area = coeff*ellipse_area
    x_new = x/np.sqrt(ellipse_area)
    y_new = y/np.sqrt(ellipse_area)
    p_new = p/np.sqrt(ellipse_area)
    e_new = e
    
    return x_new, y_new, e_new, p_new



def DiffEqSol(e,p):
    
    def Second_order(x, phi, p):
        return [x[1], 1/p - x[0]]
    
    x0 = [(1+e)/p, 0]
    phi = np.linspace(0, 6*np.pi, 300)
    sol = odeint(Second_order, x0, phi, args=(p,))
    
    solution = sol[:,0]
    
    return phi, solution
    