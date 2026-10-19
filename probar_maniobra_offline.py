"""Simulación de la máquina + ControlYaw real extraído sin importar hardware."""
import ast
from pathlib import Path
import numpy as np
from banco_headless import Banco, extraer
from maniobra_evasion import ManiobraEvasion, wrap


def constantes():
    tree = ast.parse(Path('Prueba_ruta_carrot.py').read_text(encoding='utf-8'))
    cfg = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            try: val = ast.literal_eval(n.value)
            except (ValueError, TypeError): continue
            for name in n.targets:
                if isinstance(name, ast.Name): cfg[name.id] = val
    return cfg


def crear():
    c = constantes()
    return ManiobraEvasion(
        holgura=c['EVASION_HOLGURA'], desvio_max=c['EVASION_DESVIO_MAX'],
        radio_influencia=c['EVASION_RADIO_INF'], radio_seguro=c['EVASION_RADIO_SEG'],
        cola=c['EVASION_COLA'], compromiso_min=c['EVASION_COMPROMISO_MIN'],
        v_max=min(c['V_MAX'],c['APROX_V_MAX']),
        a_max=min(c['A_LAT_MAX'],c['A_TAN_MAX'],c['APROX_A_MAX']),
        t_min=c['APROX_T_MIN'], tol_pos=c['APROX_TOL'],
        tol_yaw=np.radians(c['YAW_TOL_APROX']), edad_max=c['EVASION_VISTA_MAX_S'],
        espera_vista_max=c['MANIOBRA_ESPERA_VISTA_S'],
        espera_max=c['YAW_T_MAX_APROX'], margen_lado=c['EVASION_MARGEN_LADO'])


def simular(y_obst=0., sin_vista=False, lado_ocupado=False, corte=False):
    m = crear()
    c = constantes()
    pos, yaw = np.zeros(2), [0.]
    scope = extraer('Prueba_ruta_carrot.py', ['ControlYaw','_wrap'],
                    {**c, '_yaw_mundo_del_mocap':lambda:yaw[0]})
    yc = scope['ControlYaw']()
    yc.fijar(0.)
    yc.desfase = 0.
    obst = np.array([[.2,y_obst]])
    m.iniciar(pos, pos, (1,0), obst, [1], 0.)
    posiciones, yaws, lados, estados = [pos.copy()], [0.], [], []
    dt=.02
    vision_cortada=False
    for k in range(1,15001):
        t=k*dt
        rel=obst-pos
        ang=np.arctan2(rel[:,1],rel[:,0])
        visible=np.any((np.abs(wrap(ang-yaw[0]))<np.radians(35))
                       & (np.linalg.norm(rel,axis=1)<.8))
        if lado_ocupado and m.estado=='VERIFICAR_LADO': visible=True
        vista={'ts':t, 'valida':not sin_vista, 'bloqueada':bool(visible)}
        if corte and m.estado=='AVANCE_LATERAL' and t-m.desde>2:
            vision_cortada=True
        if vision_cortada: vista['valida']=False
        estado_previo, desde_previo = m.estado, m.desde
        salida=m.paso(pos,yaw[0],t,vista,obst,(pos[0],0.),
                      yaw_estable=abs(yc.vel)<np.radians(.5))
        if m.estado in ('AVANCE_LATERAL','AVANCE_PARALELO','RETORNO'):
            assert vista['valida'] and 0 <= t-vista['ts'] <= m.edad_max
            assert not vista['bloqueada']
            if m.estado != estado_previo: assert vista['ts'] > desde_previo
        if salida is None: break
        pos, objetivo=salida
        deg,_=yc.paso(objetivo,dt,rate_max=min(c['YAW_RATE_MAX'],c['YAW_RATE_APROX']),
                      acel_max=min(c['YAW_ACEL_MAX'],c['YAW_ACEL_APROX']))
        yaw[0]=np.radians(deg)
        posiciones.append(pos.copy()); yaws.append(yaw[0])
        lados.append(m.lado); estados.append(m.estado)
        if m.estado=='BLOQUEADA': break
    return m, np.array(posiciones), np.unwrap(yaws), np.array(lados), estados


def main():
    b=Banco()
    for lateral in (0.,-.001,.001,-.1,.1):
        m,p,y,l,e=simular(lateral)
        print('SECUENCIA:', ' > '.join(m.historial))
        b.comprobar(f'centro {lateral} sin bloqueo', m.estado=='RUTA', 'parada indefinida',m.estado)
        cambios=int(np.count_nonzero(np.diff(l)))
        b.comprobar(f'centro {lateral} cero cambios lado', cambios==0,0,cambios)
        esperado=['RUTA','GIRO_LADO','VERIFICAR_LADO','AVANCE_LATERAL',
                  'GIRO_RUTA','VERIFICAR_RUTA','AVANCE_PARALELO',
                  'GIRO_RETORNO','VERIFICAR_RETORNO','RETORNO','GIRO_FINAL','RUTA']
        b.comprobar(f'centro {lateral} orden',m.historial==esperado,len(esperado),len(m.historial))
        v=np.diff(p,axis=0)/.02
        a=np.diff(v,axis=0)/.02
        w=np.degrees(np.diff(y))/.02
        aw=np.diff(w)/.02
        b.comprobar(f'centro {lateral} rapidez', np.max(np.linalg.norm(v,axis=1))<=m.v_max+1e-6,
                    m.v_max, round(float(np.max(np.linalg.norm(v,axis=1))),6))
        b.comprobar(f'centro {lateral} aceleracion', np.max(np.linalg.norm(a,axis=1))<=m.a_max+1e-6,
                    m.a_max,round(float(np.max(np.linalg.norm(a,axis=1))),6))
        b.comprobar(f'centro {lateral} yaw suave', max(abs(w))<=12+1e-6 and max(abs(aw))<=30+1e-6,
                    (12,30),(round(max(abs(w)),4),round(max(abs(aw)),4)))
        giros=[i for i,s in enumerate(e) if s.startswith(('GIRO_','VERIFICAR_'))]
        desplaz=max((np.linalg.norm(p[i+1]-p[i]) for i in giros),default=0)
        b.comprobar(f'centro {lateral} quieto al girar', desplaz<1e-6,0,desplaz)
    for nombre,kwargs in [('sin imagen',{'sin_vista':True}),
                          ('lado ocupado',{'lado_ocupado':True}),
                          ('perdida durante avance',{'corte':True})]:
        m,p,_,l,_=simular(**kwargs)
        b.comprobar(nombre+' no presume libre',m.estado=='BLOQUEADA',0,m.estado)
        if kwargs.get('corte'):
            b.comprobar('perdida frena y espera antes de bloquear',
                        m.historial[-3:]==['FRENANDO','ESPERA_VISTA','BLOQUEADA'],
                        'FRENANDO > ESPERA_VISTA > BLOQUEADA',m.historial)
            pico=float(np.max(np.linalg.norm(np.diff(p,n=2,axis=0)/.02**2,axis=1)))
            b.comprobar('frenado por perdida suave',pico<=m.a_max+1e-6,m.a_max,pico)
        else:
            b.comprobar(nombre+' sin avance',np.max(abs(p))==0,0,float(np.max(abs(p))))
    m=crear(); m.iniciar((0,0),(0,0),(1,0),[[.2,0]],[1],0)
    m.paso((0,0),m.yaw_lado,1,{'ts':.5,'valida':True,'bloqueada':False},[],(0,0))
    m.paso((0,0),m.yaw_lado,1.1,{'ts':.5,'valida':True,'bloqueada':False},[],(0,0))
    b.comprobar('imagen anterior al giro rechazada',m.estado=='VERIFICAR_LADO',.5,m.desde)
    m=crear(); m.iniciar((0,0),(0,0),(1,0),[[.2,0]],[1],0)
    m.yaw_obj=m.yaw_ruta
    m.cambiar('GIRO_RUTA',0.)
    pos=np.array([.03,.08])
    t=m.espera_max+.1
    vista={'ts':t-.4,'valida':True,'bloqueada':False}
    salida=m.paso(pos,m.yaw_obj+np.pi/2,t,vista,[],(0,0),yaw_estable=False)
    b.comprobar('giro sin converger reintenta una vez en el mismo estado',
                m.estado=='GIRO_RUTA' and m.reintentos==1 and m.desde==t
                and np.array_equal(m.fijo,pos) and np.array_equal(salida[0],pos)
                and m.motivo.startswith('Reintento: '),
                'GIRO_RUTA, 1 reintento, ancla actual',
                (m.estado,m.reintentos,m.fijo.tolist()))
    b.comprobar('diagnostico del giro antes de reanclar',
                abs(m.diag['err_yaw_deg']-90.)<1e-9
                and abs(m.diag['err_pos_cm']-np.linalg.norm(pos)*100)<1e-9
                and m.diag['yaw_estable'] is False
                and abs(m.diag['edad_vista_s']-.4)<1e-9,
                '90 grados, distancia al ancla previa, False, 0.4 s',m.diag)
    t+=m.espera_max+.1
    m.paso(pos,m.yaw_obj+np.pi/2,t,{'ts':t-.4,'valida':True,'bloqueada':False},
           [],(0,0),yaw_estable=False)
    b.comprobar('segundo plazo agotado bloquea con diagnostico',
                m.estado=='BLOQUEADA' and m.reintentos==1
                and 'yaw' in m.motivo and 'pos' in m.motivo,
                'BLOQUEADA con yaw y pos',(m.estado,m.motivo))
    probar_espera_vista(b)
    return b.terminar()


def probar_espera_vista(b):
    # Alcanzar cada tramo por transiciones reales, siempre con imagen fresca.
    for avance in ('AVANCE_LATERAL','AVANCE_PARALELO','RETORNO'):
        for perdida in ('ausente','invalida','vieja'):
            for respuesta in ('libre','bloqueada','timeout'):
                nombre=f'{avance} {perdida} {respuesta}'
                m=crear()
                pos=np.zeros(2)
                m.iniciar(pos,pos,(1,0),[[.2,0]],[1],0.)
                t=0.
                for _ in range(10000):
                    t+=.02
                    salida=m.paso(pos,m.yaw_obj,t,
                                  {'ts':t,'valida':True,'bloqueada':False},[],(.63,0))
                    if salida is not None: pos=salida[0]
                    if m.estado==avance and t-m.desde>=1.: break
                b.comprobar(nombre+' alcanza tramo',m.estado==avance,avance,m.estado)
                destino=m.tramo.fin.copy()
                t+=.02
                vista=(None if perdida=='ausente' else
                       {'ts':t-m.edad_max-.1 if perdida=='vieja' else t,
                        'valida':perdida!='invalida','bloqueada':False})
                pos=m.paso(pos,m.yaw_obj,t,vista,[],(.63,0))[0]
                b.comprobar(nombre+' frena',m.estado=='FRENANDO','FRENANDO',m.estado)
                b.comprobar(nombre+' diagnostico avance',
                            m.diag['edad_vista_s']==(None if vista is None else t-vista['ts'])
                            and m.diag['err_pos_cm']==0.,'edad actual y error cero',m.diag)
                for _ in range(1000):
                    if m.estado!='FRENANDO': break
                    t+=.02
                    pos=m.paso(pos,m.yaw_obj,t,None,[],(.63,0))[0]
                b.comprobar(nombre+' espera',m.estado=='ESPERA_VISTA','ESPERA_VISTA',m.estado)
                fijo=pos.copy(); yaw=m.yaw_obj; desde=m.desde
                # Ni una vista libre anterior al final del freno autoriza avanzar.
                for retraso in (.02,.2):
                    t=desde+retraso
                    pos=m.paso(pos,yaw,t,{'ts':desde,'valida':True,'bloqueada':False},[],(.63,0))[0]
                    b.comprobar(nombre+f' inmovil sin vista nueva {retraso}',
                                m.estado=='ESPERA_VISTA' and np.array_equal(pos,fijo)
                                and m.yaw_obj==yaw,'ESPERA_VISTA fija',m.estado)
                    b.comprobar(nombre+f' diagnostico espera {retraso}',
                                abs(m.diag['edad_vista_s']-retraso)<1e-9,
                                retraso,m.diag['edad_vista_s'])
                if respuesta=='timeout':
                    t=desde+m.espera_vista_max-.001
                    m.paso(pos,yaw,t,None,[],(.63,0))
                    b.comprobar(nombre+' respeta plazo',m.estado=='ESPERA_VISTA','ESPERA_VISTA',m.estado)
                    t+=.1
                    m.paso(pos,yaw,t,None,[],(.63,0))
                    motivo=f'Visión ausente durante el avance ({t-desde:.1f} s)'
                    b.comprobar(nombre+' motivo exacto',m.estado=='BLOQUEADA' and m.motivo==motivo,
                                motivo,m.motivo)
                else:
                    t+=.02
                    pos=m.paso(pos,yaw,t,{'ts':t,'valida':True,
                                         'bloqueada':respuesta=='bloqueada'},[],(.63,0))[0]
                    if respuesta=='bloqueada':
                        b.comprobar(nombre+' bloquea quieto',m.estado=='BLOQUEADA'
                                    and m.motivo=='Obstáculo nuevo durante el avance'
                                    and np.array_equal(pos,fijo),'BLOQUEADA',m.estado)
                    else:
                        b.comprobar(nombre+' retoma mismo destino desde fijo',
                                    m.estado==avance and np.array_equal(m.tramo.fin,destino)
                                    and np.array_equal(m.tramo.inicio,fijo)
                                    and m.historial[-3:]==['FRENANDO','ESPERA_VISTA',avance],
                                    avance,m.historial[-3:])
                        for _ in range(10000):
                            t+=.02
                            pos=m.paso(pos,yaw,t,{'ts':t,'valida':True,'bloqueada':False},[],(.63,0))[0]
                            if m.estado!=avance: break
                        siguiente={'AVANCE_LATERAL':'GIRO_RUTA',
                                   'AVANCE_PARALELO':'GIRO_RETORNO','RETORNO':'GIRO_FINAL'}[avance]
                        b.comprobar(nombre+' llega al destino',m.estado==siguiente
                                    and np.linalg.norm(pos-destino)<1e-9,siguiente,m.estado)


if __name__=='__main__': raise SystemExit(main())
