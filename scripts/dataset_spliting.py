from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import random as rd
import numpy as np
import os
def get_dataframe_with_category(dic_category,filename):
    super_df=pd.read_csv(filename)
    super_df=super_df[~super_df.categorias.isnull()]
    categories=super_df.categorias.to_list()
    new_categories=[]
    new_categories_2=[]
    for cat in categories:
        new_cat={}
        for super_category,subcategories in dic_category.items():
            for subcategory in subcategories:
                if subcategory in cat:
                    new_cat.setdefault(super_category,0)
                    new_cat[super_category]+=1
        if len(new_cat)==0:
            new_categories.append(None)
        else:
            values=list(new_cat.values())
            if np.std(values)==0 and len(new_cat)>1:
                c=rd.choice(list(new_cat.keys()))
                new_categories.append(c)
            else:
                max_val=0
                max_cat=""
                for i,j in new_cat.items():
                    if j>max_val:
                        max_cat=i
                new_categories.append(max_cat)

    super_df["super_category"]=new_categories
    super_df=__min_max_sclation_df(super_df)
    merge_df=__merge_df(super_df)
    merge_df["metrica_exito"]=0.4*merge_df.max_min_estrellas+0.4*merge_df.max_min_resenas+merge_df.Sentiment*0.2
    merge_df=__basic_clean(merge_df)
    merge_df=__convert_days_to_bynary(merge_df)
    merge_df=__split_morning_after_nig(merge_df)
    __convert_24_horas(merge_df)
    
    return merge_df

def __merge_df(df):
    import pickle
    file=open("../serialized_data/final_mean_sentiment","rb")
    df_sentiment=pickle.load(file)
    file.close()
    df=pd.merge(df,df_sentiment,on="negocio_id")
    return df

def __min_max_sclation_df(daf):
    df=__delete_by_ir(daf)
    
    trans=MinMaxScaler()
    data=df[["estrellas","cantidad_reseñas"]]
    data=trans.fit_transform(data)
    data_df=pd.DataFrame(data)
    data_df.columns=["max_min_estrellas","max_min_resenas"]
    concat_df=pd.concat((df,data_df),axis=1)
    return concat_df 

def __delete_by_ir(df):
    q1=np.quantile(df.cantidad_reseñas,0.25)
    q3=np.quantile(df.cantidad_reseñas,0.75)
    iqr=q3-q1
    min_limit=q1-1.5*iqr
    max_limit=q3+1.5*iqr
    f_df=df[df.cantidad_reseñas>min_limit]
    f_df=f_df[f_df.cantidad_reseñas<max_limit]
    f_df=f_df.reset_index(drop=True)
    return f_df

def get_dataframes_by_categories(dic_category,filename):
    groups=get_dataframe_with_category(dic_category,filename).groupby("super_category",as_index=False)
    name_groups=groups.groups.keys()
    df_dict={}
    if not(os.path.isdir("../files/datasets/")):
        os.makedirs('../files/datasets', exist_ok=True)

    for i in name_groups:
        df_dict[i]=groups.get_group(i).reset_index(drop=True)
        df_dict[i].drop(["super_category"],axis=1,inplace=True)
    del groups
    for clave,valor in df_dict.items():
        valor.to_csv(f"../files/datasets/csv_{clave}.csv")
    return 0


def __basic_clean(df):
    df_1=df[df.abierto==1]
    import ast
    new_list=[]
    for i in df_1.parqueo.to_list():
        if i is np.nan or i==None or i=="None" or len(i)==0:
            new_list.append(None)
        else:
            d=ast.literal_eval(i)
            val=d.get("estacionamiento",None)
            if val==None:
                new_list.append(None)
            else:
                new_list.append(val)
    new_list=[]
    for i in df_1.rango_precios:
        if i is np.nan or i==None or i=="None":
            new_list.append(np.nan)
        else:
            new_list.append(float(i))
    df_1.rango_precios=new_list
    df_1.parqueo=__convert_column_to_binary(new_list)
    df_1.acepta_tarjeta_credito=__convert_column_to_binary(df_1.acepta_tarjeta_credito)
    df_1.tiene_parqueo_bicicletas=__convert_column_to_binary(df_1.tiene_parqueo_bicicletas)
    return df_1

def __convert_days_to_bynary(df):
    week_days=["atencion_lunes","atencion_martes","atencion_miercoles","atencion_jueves","atencion_viernes"]
    weekend_days=["atencion_sabado","atencion_domingo"]

    

    with_out_na=df[week_days+weekend_days].dropna(how="all")

    df_without_nans=pd.merge(df,with_out_na,left_index=True, right_index=True)
    df_without_nans=df_without_nans.reset_index(drop=True)
    week_days=["atencion_lunes_y","atencion_martes_y","atencion_miercoles_y","atencion_jueves_y","atencion_viernes_y"]
    weekend_days=["atencion_sabado_y","atencion_domingo_y"]
    d_w={"week_days":week_days,"weekend_days":weekend_days}
    for i,j in d_w.items():
        new_days=df_without_nans[j].any(axis=1)
        df_without_nans[i]=__convert_column_to_binary(new_days)

    return df_without_nans
        

def __convert_column_to_binary(column):
    l=[]
    for i in column:
        if i is np.nan or i==None or i=="None":
            l.append(None)
        else:
            if i=="True" or i==True:
                l.append(1)
            else:
                l.append(0)
    return l

def __split_morning_after_nig(df):
    week_days=["atencion_lunes_y","atencion_martes_y","atencion_miercoles_y","atencion_jueves_y","atencion_viernes_y"]
    weekend_days=["atencion_sabado_y","atencion_domingo_y"]
    df_limpio=df[week_days+weekend_days].apply(__convert_range_to_numeric)
    df.drop(columns=week_days+weekend_days,inplace=True)
    df=pd.concat((df,df_limpio),axis=1)
    df_tuplas=df_limpio.apply(__get_max_interval,axis=1)
    df["max_interval"]=df_tuplas
    return __built_df_with_schedule(df)

def __built_df_with_schedule(df):
    #1 isMorning
    #2 isAfter
    #3 isNight
    #4 isM & isA
    #5 isA & isN
    #6 isM & isA & isN
    week_days=["atencion_lunes_y","atencion_martes_y","atencion_miercoles_y","atencion_jueves_y","atencion_viernes_y"]
    weekend_days=["atencion_sabado_y","atencion_domingo_y"]
    
    for day in week_days+weekend_days:
        l=[]
        for ran in df[day]:
            cat=None
            if ran is np.nan or ran==None or ran=="None":
                l.append(cat)
            else:
                min_i=ran[0]
                max_i=ran[-1]
                if 5<=min_i<=12 or 5<=max_i<=12:
                    cat=1
                if 12<min_i<=18 or 12<max_i<=18:
                    cat=2
                if 18<min_i<=23 or(18<max_i<=23 or 0<=max_i<5):
                    cat=3
                if 5<=min_i<=12 and 12<max_i<=18:
                    cat=4
                if 12<min_i<=18 and (18<max_i<=23 or 0<=max_i<5):
                    cat=5
                if 5<=min_i<=12 and (18<max_i<=23 or 0<=max_i<5):
                    cat=6
                if 0.0==min_i and 0==max_i:
                    cat=6
                l.append(cat)
        df[f"horario_{day}"]=l
    return df

def __convert_24_horas(df):
    df.abierto_24_horas=df["max_interval"].apply(__is_24_horas)


def __is_24_horas(tuple_range):
    if tuple_range==(0.0,0.0):
        return 1
    return 0

def __get_max_interval(serie):
    min_val=30
    max_val=-30
    for i in serie:
        if i is np.nan or i==None:
            pass
        else:
            begin=i[0]
            end=i[-1]
            if min_val>begin:
                min_val=begin
            if 0<=end<=11:
                max_val=end
            elif max_val<end:
                max_val=end
    return (min_val,max_val)

def __convert_range_to_numeric(series):
    lista=[]
    convert_is_decimal=lambda x: float(x[0]+".5") if int(x[-1])!=0 else float(x[0])
    for num_str in series:  
        if num_str is np.nan or num_str==None:
            lista.append(None)
        else:
            num_str_s=num_str.split("-")
            begin_split=num_str_s[0].split(":")
            end_split=num_str_s[-1].split(":")
            lista.append((convert_is_decimal(begin_split),convert_is_decimal(end_split)))
    return lista
