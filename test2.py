import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import warnings
import ast
import pickle
import tempfile
from io import BytesIO
from datetime import datetime

# 创建基面1和基面2
def show_base1():
    try:
        # 在这里放置您的Streamlit应用程序逻辑
        # 输入文件名======================================================================
        data_file = '必要数据'
        # 2、获取低频数据和高频数据的中文表头
        file_name = r'必要数据' \
                    r'/低频数据和高频数据表头20230823.xlsx'
        df_sta_name = pd.read_excel(file_name,sheet_name='低频数据') # 读取低频数据
        df_ts_name = pd.read_excel(file_name,sheet_name='高频数据')
        # 2、首先读取低频数据和高频数据===========================
        # 第一次读取Excel数据并进行处理
        # 缓存数据： 使用st.cache装饰器将数据加载函数标记为缓存，以确保处理过的数据只在第一次调用时被加载。
        # 使用缓存： Streamlit的缓存机制将确保数据只在第一次调用load_data()函数时加载，并在之后的请求中直接返回缓存的数据。
        # 再次读取时直接加载： 在之后的请求中，直接加载处理后的数据，而不必重新读取和处理原始的Excel数据。
        # @st.cache_data
        # def load_sta_data():
        #     df_sta = pd.read_pickle(os.path.join(data_file, 'ZI 低频-脱敏发出20230823.pkl'))
        #     return df_sta
        # @st.cache_data
        def load_ts_data():
            df_ts = pd.read_pickle(os.path.join(data_file, 'ZI 高频-脱敏发出20230823.pkl'))
            return df_ts
        # @st.cache_data
        def load_fiter_data_ts(filtered_data):
            # 5、找出低频数据和高频数据重复的炉次号
            number = df_ts['标识列'].unique()  # 得到1052个炉次的熔炼号
            number = pd.DataFrame(number, columns=['标识列'])
            data_sta = pd.merge(number, filtered_data, on='标识列')  # 低频数据和高频炉次号合并
            # 6、高频数据每一炉次画原始曲线图，并在图上标记出来几个时刻点
            # 6.1 先把每个炉次的时间顺序调过来
            Data_ts_len = []   # # 获取每一炉次的采样点个数
            Data_ts = []    # 获取每一炉次的数据
            for i in range(data_sta.shape[0]):
                every_data = df_ts[df_ts['标识列'] == data_sta.loc[i,'标识列']].reset_index()
                every_data = every_data.sort_values(by='index', ascending=False)  # 按照时间逆序排序
                every_data['index'] = np.arange(0, every_data.shape[0], 1)  # 更改index列
                every_data = every_data.reset_index(drop=True)  # 重置索引并丢弃旧索引
                Data_ts_len.append(every_data.shape[0]) # 每一炉次的原始吹炼时间长度
                Data_ts.append(every_data.apply(pd.to_numeric, errors = 'raise'))
            return Data_ts

        # 保存文件到本地路径的函数
        def save_file_to_local_path_sta(file, local_path):
            # 将 DataFrame 转换为 CSV 格式的字节流
            csv = file.to_csv(index=False).encode('utf-8-sig') # ,encoding='utf-8-sig'
            # 将字节流转换为 BytesIO 对象
            csv_io = BytesIO(csv)
            # 提供下载链接
            st.download_button(label="保存低频数据到本地路径", data=csv_io,
                               file_name=local_path, mime='text/csv')

        def save_file_to_local_path_ts(file, local_path):
            np.save(local_path,file)
            # 读取 .npy 文件并转换为字节流
            with open(local_path, "rb") as f:
                npy_data = f.read()
            st.download_button(
                label="保存高频数据到本地路径",
                data=npy_data,
                file_name=local_path,
                mime="application/octet-stream")

            # 标题
        st.title("转炉质量根因分析模型离线仿真器")
        # 创建一个二级标题
        st.subheader("1-加载数据")
        uploaded_file_sta = st.file_uploader("上传低频数据", type=['pkl','csv','xlsx'],
                                             key="sta_file_uploader")  # ,'csv'
        # 检查是否有文件上传
        if uploaded_file_sta is not None:
            # 读取上传的低频数据
            df_sta = pd.read_pickle(uploaded_file_sta)
            # 3、替换掉脱敏数据中的代码（列名），使用真实中文变量名
            df_sta.columns = df_sta_name.iloc[0, :]
            if 'TSO检测碳含量' in df_sta:
                df_sta["TSO检测碳含量"] = pd.to_numeric(df_sta["TSO检测碳含量"], errors="coerce")
            st.write("低频数据载入成功")
            st.write(df_sta)
            # 第一行：开始时间-结束时间选择器,出钢钢号下拉菜单
            col1, col2, col3, col4 = st.columns(4)
            # 开始时间选择器
            df_sta['日期_new'] = pd.to_datetime(df_sta.日期, format='%Y%m%d').dt.date  # 将日期列转换为日期类型
            start_date_default = pd.to_datetime(df_sta.日期, format='%Y%m%d').min()
            start_date = col1.date_input('开始日期',
                                         value=start_date_default,
                                         min_value=df_sta['日期_new'].min(),
                                         max_value=df_sta['日期_new'].max()
                                         )
            # 结束时间选择器
            end_date_default = pd.to_datetime(df_sta.日期, format='%Y%m%d').max()
            end_date = col2.date_input('结束日期',
                                       value=end_date_default,
                                       min_value=df_sta['日期_new'].min(),
                                       max_value=df_sta['日期_new'].max()
                                       )
            # 选择出钢钢号
            steel_no = ['待增加该数据项']
            # 添加“全选”选项
            select_all = col4.checkbox("全选", key="select_all")
            # 根据全选选项设置multiselect的默认值
            steel = col3.multiselect("选择初始出钢钢号", steel_no, default=steel_no if select_all else [])
            col1, col2, col3, col4 = st.columns(4)
            # 选择最终出钢钢号
            steel_final_no = list(set(df_sta.钢种号.tolist()))
            steel_final_all1 = col2.checkbox("全选", key="steel_final_all1")
            steel_final = col1.multiselect("选择最终出钢钢号", steel_final_no,
                                           default=steel_final_no if steel_final_all1 else [])
            # 选择转炉号
            # 取某一列的数字的第一位，作为新的一列
            df_sta['转炉号'] = df_sta['标识列'].astype(str).str[0].astype(int)
            zl_no = list(set(df_sta.转炉号.tolist()))
            select_all2 = col4.checkbox("全选", key="select_all2")
            zl = col3.multiselect("选择转炉号", zl_no, default=zl_no if select_all2 else [])

            col1, col2, col3, col4 = st.columns(4)
            # 选择班号
            class_no = list(set(df_sta.班组.tolist()))
            select_all1 = col2.checkbox("全选", key="select_all1")
            class1 = col1.multiselect("选择班号", class_no, default=class_no if select_all1 else [])
            # 选择组号
            group_no = ['待增加该数据项']
            group_all1 = col4.checkbox("全选", key="group_all1")
            group1 = col3.multiselect("选择组号", group_no, default=group_no if group_all1 else [])

            col1, col2 = st.columns(2)
            # 选择制造命令号
            Manufac_order_no = ['待增加该数据项']
            Manufac_order_all1 = col2.checkbox("全选", key="Manufac_order_all1")
            Manufac_order1 = col1.multiselect("选择制造命令号",
                                              Manufac_order_no,
                                              default=Manufac_order_no if Manufac_order_all1 else [])

            # # 勾选框
            # load_high_frequency_data = st.checkbox("加载高频数据")
            # 根据用户选择加载相应的数据
            filtered_data = df_sta[(df_sta['日期_new'] >= start_date) & (df_sta['日期_new'] <= end_date) &
                                   df_sta['钢种号'].isin(steel_final) & df_sta['班组'].isin(class1) &
                                   df_sta['转炉号'].isin(zl)]
            # 本地保存路径
            local_path1 = st.text_input("低频数据文件保存（文件名.csv）:",
                                        '模块1输出-低频-炉次数为{}_{}-{}-钢号{}-班组{}-转炉号{}.csv'.format(
                                            filtered_data.shape[0], start_date, end_date, steel_final,
                                            class1, zl
                                        )
                                        )
            # 按钮 # 保存文件到本地路径
            save_file_to_local_path_sta(filtered_data, local_path1)
            # 勾选框
            load_high_frequency_data = st.checkbox("加载高频数据")
            if load_high_frequency_data:
                # 文件上传框
                uploaded_file_ts = st.file_uploader("上传高频数据", type=['pkl', 'csv', 'xlsx', 'rar'],
                                                    key="ts_file_uploader")  # ,'csv'
                if uploaded_file_ts is not None:
                    # 读取上传的高频数据
                    df_ts = load_ts_data()
                    # 3、替换掉脱敏数据中的代码（列名），使用真实中文变量名
                    df_ts.columns = df_ts_name.iloc[0, :]
                    st.write("高频数据载入成功")
                    # st.write(df_sta)
                    # 4、处理高频数据中的时间列，将其转换为与低频数据时刻对应的时间格式
                    df_ts['时刻'] = df_ts['时间列'].map(
                        lambda x: time.strftime("%Y%m%d%H%M%S", time.localtime(round(x / 1000))),
                        na_action='ignore')
                    ts_list = load_fiter_data_ts(filtered_data)
                    local_path2 = st.text_input("高频数据文件保存（文件名.npy）:",
                                                '模块1输出-高频-炉次数为{}_{}-{}-钢号{}-班组{}-转炉号{}.npy'.format(
                                                    len(ts_list), start_date, end_date, steel_final,
                                                    class1, zl
                                                )
                                                )
                    # 保存文件到本地路径
                    save_file_to_local_path_ts(ts_list, local_path2)
                else:
                    # 如果没有文件上传，显示提示
                    st.write("请上传文件以载入数据")
            # 按钮
            load_data_button = st.button("查看数据")
            col5, col6 = st.columns([1, 1])
            # 如果加载高频数据勾选框没选中，点击了加载数据，则仅读取低频数据，不加载高频数据
            if load_data_button:
                print("加载数据按钮被点击了")
                if not load_high_frequency_data:
                    print("未加载高频数据")
                    col5.text("加载低频数据,共{}炉次".format(filtered_data.shape[0]))
                    col5.write(filtered_data)
                    col5.text("低频数据加载完成！共{}炉次".format(filtered_data.shape[0]))
                    col6.text("不加载高频数据")
                    # # 在Streamlit应用中使用保存文件的函数
                    # if save_data_button_sta:
                    #     # 保存文件到本地路径
                    #     save_file_to_local_path_sta(filtered_data, local_path1)
            # 如果加载高频数据勾选框被选中，并点击了加载数据，则读取低频数据和高频数据
            if load_high_frequency_data:
                if load_data_button:
                    print("加载数据按钮被点击了")
                    col5.text("加载低频数据,共{}炉次".format(filtered_data.shape[0]))
                    # 根据用户选择加载相应的数据
                    col5.write(filtered_data)
                    col5.text("低频数据加载完成！共{}炉次".format(filtered_data.shape[0]))
                    col6.text("加载高频数据,共{}炉次".format(len(ts_list)))
                    # 在界面上显示每个DataFrame
                    for idx, df in enumerate(ts_list):
                        # col6.write('标识列 {}:'.format(df['标识列'].iloc[0]))
                        col6.dataframe(df)
                        col6.write('---')  # 用分隔线分隔每个DataFrame
                    col6.text("高频数据加载完成！共{}炉次".format(len(ts_list)))
        else:
            # 如果没有文件上传，显示提示
            st.write("请上传文件以载入数据")




    except Exception as e:
        # 捕获异常并在控制台中记录
        print(f"发生异常: {e}")

def show_base2():
    pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))


    file = "必要数据\8_606炉次_241变量.xlsx"
    file_name_2 = "必要数据\8_606炉次_变量名.csv"

    def load_sta_data(file):
        data = pd.read_excel(file)
        return data

    def download_result(data1, formatted_time):
        df = data1

        # 将 DataFrame 转换为 CSV 格式的字节流
        csv = df.to_csv(index=False).encode('utf-8-sig')

        # 将字节流转换为 BytesIO 对象
        csv_io = BytesIO(csv)

        # 提供下载链接
        st.download_button(label="下载文件至本地路径", data=csv_io, file_name='变量选择后文件_{}.csv'.format(
            formatted_time), mime='text/csv')

    st.title("转炉质量根因分析模型离线仿真器")
    # 创建一个二级标题
    st.subheader("4-变量选择模块（自定义变量）")

    uploaded_file_sta = st.file_uploader("上传数据", type=['xlsx','csv'])  # ,'csv'
    # 检查是否有文件上传
    if uploaded_file_sta is not None:
        # 读取上传的低频数据
        st.success("数据载入成功")
        # dirname = os.path.dirname(file)
        data = load_sta_data(uploaded_file_sta)
        data_C = []
        data['铁水C'] = np.nan_to_num(data['铁水C'])
        data['铁水C_new'] = data['铁水C']
        for i in range(data.shape[0] - 1):
            if (data.铁水C.iloc[i] != 0 and data.铁水C.iloc[i + 1] == 0):
                data.铁水C_new.iloc[i + 1] = data.铁水C_new.iloc[i]
        for i in range(data.shape[0]):
            if (data.铁水C.iloc[i] != 0):
                data_C.append(data.铁水C.iloc[i])
        median_C = np.median(np.array(data_C))
        for i in range(data.shape[0]):
            if (data.铁水C_new.iloc[i] == 0):
                data.铁水C_new.iloc[i] = median_C

        # data['铁水C'] = np.nan_to_num(data['铁水C'])
        # #设置随机种子
        # np.random.seed(42)
        # # 定义均值和标准差
        # C_mean = data['铁水C'][data['铁水C'] != 0].mean()
        # C_std = np.std(data['铁水C'][data['铁水C'] != 0])
        # # 生成服从正态分布的随机噪声
        # noise = np.random.normal(0, 10, size=len(data['铁水C'][data['铁水C'] == 0]))
        # # 填充数据
        # C_data = C_mean + noise
        # # 将铁水C中的0值用随机生成的数字进行替换
        # data['铁水C'][data['铁水C'] == 0] = C_data

        data['收得率'] = (data['铁水加入量'] * (
                    1 - data['铁水C_new'] / 10000 - data['铁水Si'] / 10000 - 0.5 * data['铁水Mn'] / 10000) + \
                          data['废钢量'] * (1 - 0.0018 - 0.002 - 0.0052)) / (data['铁水加入量'] + data['废钢量'])
        data['TSO温度减目标温度差值'] = data['TSO检测温度'] - data['目标温度']
        data['氧枪高度吹炼前中期均值'] = (data['氧枪高度吹炼前期均值'] + data['氧枪高度吹炼中期均值']) / 2
        data['吹炼前中期_辅料加入总量'] = data['吹炼前期_辅料加入总量_new'] + data['吹炼中期_辅料加入总量']

        # 最新组合
        data['(铁水C_new-TSC检测碳含量)/((吹炼前期耗氧量+吹炼中期耗氧量)*(铁水加入量+废钢加入量)*收得率)'] = \
            (data['铁水C_new'] - data['TSC检测碳含量']) / (
                    ((data['吹炼前期耗氧量'] + data['吹炼中期耗氧量']) / 3600) * (data['铁水加入量'] + data['废钢量']) *
                    data['收得率'])
        data['铁水Si/(吹炼前期耗氧量*(铁水加入量+废钢加入量)*收得率)'] = \
            data['铁水Si'] / ((data['吹炼前期耗氧量'] / 3600) * (data['铁水加入量'] + data['废钢量']) * data['收得率'])

        data['吹炼后期耗氧量/((TSO检测温度-TSC检测温度)*(铁水加入量+废钢量)*收得率)'] = (data[
                                                                                             '吹炼后期耗氧量'] / 3600) / (
                                                                                                (data['TSO检测温度'] -
                                                                                                 data[
                                                                                                     'TSC检测温度']) * (
                                                                                                            data[
                                                                                                                '铁水加入量'] +
                                                                                                            data[
                                                                                                                '废钢量']) *
                                                                                                data['收得率'])
        data['(吹炼前期耗氧量+吹炼中期耗氧量)/（(TSC检测温度-铁水温度)*(铁水加入量+废钢量)*收得率）'] = ((data[
                                                                                                            '吹炼前期耗氧量'] +
                                                                                                        data[
                                                                                                            '吹炼中期耗氧量']) / 3600) / (
                                                                                                              (data[
                                                                                                                   'TSC检测温度'] -
                                                                                                               data[
                                                                                                                   '铁水温度']) * (
                                                                                                                          data[
                                                                                                                              '铁水加入量'] +
                                                                                                                          data[
                                                                                                                              '废钢量']) *
                                                                                                              data[
                                                                                                                  '收得率'])

        # 时间格式
        time_format = "%Y%m%d%H%M%S"
        # 升温速度
        data['TSO时刻-TSC时刻'] = (pd.to_datetime(data['TSO时刻'], format=time_format) -
                                   pd.to_datetime(data['TSC时刻'], format=time_format)).dt.total_seconds()
        data['TSC时刻-吹炼开始时刻'] = (pd.to_datetime(data['TSC时刻'], format=time_format) -
                                        pd.to_datetime(data['吹炼开始时刻_new'], format=time_format)).dt.total_seconds()
        data['吹炼后期升温速度'] = (data['TSO检测温度'] - data['TSC检测温度']) / (
                data['TSO时刻-TSC时刻'] * (data['铁水加入量'] + data['废钢量']) * data['收得率'])
        data['吹炼前期中期平均升温速度'] = (data['TSC检测温度'] - data['铁水温度']) / (
                data['TSC时刻-吹炼开始时刻'] * (data['铁水加入量'] + data['废钢量']) * data['收得率'])

        data['供氧强度'] = data['氧气流量吹炼全程均值'] / ((data['铁水加入量'] + data['废钢量']) * data['收得率'])
        # data['底部供气强度'] = data['底吹气体流量均值吹炼全程均值']/((data['铁水加入量']+data['废钢量'])*data['收得率'])

        data['(TSC检测温度-废钢的初始室温)/((TSC时刻-吹炼开始时刻)* 废钢加入量)'] = 0
        for i in range(data.shape[0]):
            for j in range(11):
                if (data['废钢{}加入量'.format(j + 1)].iloc[i] != 0):
                    data['(TSC检测温度-废钢的初始室温)/((TSC时刻-吹炼开始时刻)* 废钢加入量)'].iloc[i] = \
                        data['(TSC检测温度-废钢的初始室温)/((TSC时刻-吹炼开始时刻)* 废钢加入量)'].iloc[i] + \
                        (data['TSC检测温度'].iloc[i] - 25) / (
                                    data['TSC时刻-吹炼开始时刻'].iloc[i] * data['废钢{}加入量'.format(j + 1)].iloc[i])

        data['矿石加入量/((TSO检测温度-铁水温度)*(铁水加入量+废钢加入量)*收得率）'] = (data['预装_辅原料2加入量'] + data[
            '前装_辅原料2加入量'] \
                                                                                      + data['吹炼前期_辅原料2加入量'] +
                                                                                      data['吹炼中期_辅原料2加入量'] +
                                                                                      data[
                                                                                          'TSC之后_辅原料2加入量']) / ((
                                                                                                                                   data[
                                                                                                                                       'TSO检测温度'] -
                                                                                                                                   data[
                                                                                                                                       '铁水温度'])
                                                                                                                       * (
                                                                                                                                   data[
                                                                                                                                       '铁水加入量'] +
                                                                                                                                   data[
                                                                                                                                       '废钢量']) *
                                                                                                                       data[
                                                                                                                           '收得率'])
        data['Ca系辅料加入量/((TSO检测温度-铁水温度)*(铁水加入量+废钢加入量)*收得率）'] = (data['预装_辅原料1加入量'] +
                                                                                          data['前装_辅原料1加入量'] \
                                                                                          + data[
                                                                                              '吹炼前期_辅原料1加入量'] +
                                                                                          data[
                                                                                              '吹炼中期_辅原料1加入量'] +
                                                                                          data[
                                                                                              'TSC之后_辅原料1加入量']) / (
                                                                                                     (data[
                                                                                                          'TSO检测温度'] -
                                                                                                      data[
                                                                                                          '铁水温度']) * (
                                                                                                             data[
                                                                                                                 '铁水加入量'] +
                                                                                                             data[
                                                                                                                 '废钢量']) *
                                                                                                     data['收得率'])
        data['Mg系辅料加入量/((TSO检测温度-铁水温度)*(铁水加入量+废钢加入量)*收得率）'] = (data['预装_辅原料3加入量'] +
                                                                                          data['前装_辅原料3加入量'] +
                                                                                          data[
                                                                                              '吹炼前期_辅原料3加入量'] +
                                                                                          data[
                                                                                              '吹炼中期_辅原料3加入量'] +
                                                                                          data['TSC之后_辅原料3加入量'] \
                                                                                          + data['预装_辅原料8加入量'] +
                                                                                          data['前装_辅原料8加入量'] +
                                                                                          data[
                                                                                              '吹炼前期_辅原料8加入量'] +
                                                                                          data[
                                                                                              '吹炼中期_辅原料8加入量'] +
                                                                                          data[
                                                                                              'TSC之后_辅原料8加入量'] \
                                                                                          + data['预装_辅原料9加入量'] +
                                                                                          data['前装_辅原料9加入量'] +
                                                                                          data[
                                                                                              '吹炼前期_辅原料9加入量'] +
                                                                                          data[
                                                                                              '吹炼中期_辅原料9加入量'] +
                                                                                          data[
                                                                                              'TSC之后_辅原料9加入量'] \
                                                                                          + data[
                                                                                              '预装_辅原料10加入量'] +
                                                                                          data['前装_辅原料10加入量'] +
                                                                                          data[
                                                                                              '吹炼前期_辅原料10加入量'] +
                                                                                          data[
                                                                                              '吹炼中期_辅原料10加入量'] +
                                                                                          data[
                                                                                              'TSC之后_辅原料10加入量']) / (
                                                                                                     (data[
                                                                                                          'TSO检测温度'] -
                                                                                                      data['铁水温度'])
                                                                                                     * (data[
                                                                                                            '铁水加入量'] +
                                                                                                        data[
                                                                                                            '废钢量']) *
                                                                                                     data['收得率'])
        data['所需热量比值'] = (data['铁水加入量'] * ((data['TSC检测温度'] - data['铁水温度']) * 0.837)) / (
                data['废钢量'] * ((1517 - 25) * 0.699 + 272 + (data['TSC检测温度'] - 1517) * 0.837))

        # 转换整数值为时间字符串并计算时间差
        data['吹炼时长'] = (pd.to_datetime(data['氧枪提枪时刻'], format=time_format) -
                            pd.to_datetime(data['吹炼开始时刻_new'], format=time_format)).dt.total_seconds()
        # data['TSC温度减TSO温度'] = pd.to_numeric(data['TSC温度减TSO温度'], errors='coerce')
        for column in data.columns:
            data[column] = data[column].apply(
                lambda x: round(x) if isinstance(x, (int, float)) and (x > 1 or x < -1) else x)
        sta_name = pd.read_csv(file_name_2).iloc[:, 1].tolist()
        Clean_normal1 = data[(data['温度控制目标符合标签'] == 1) & (data['吹炼后期耗氧量'] != 0)]
        Clean_abnormal1 = data[(data['温度控制目标符合标签'] == 0) & (data['吹炼后期耗氧量'] != 0)]

        column = ['(铁水C_new-TSC检测碳含量)/((吹炼前期耗氧量+吹炼中期耗氧量)*(铁水加入量+废钢加入量)*收得率)',
                  '铁水Si/(吹炼前期耗氧量*(铁水加入量+废钢加入量)*收得率)',
                  '吹炼后期耗氧量/((TSO检测温度-TSC检测温度)*(铁水加入量+废钢量)*收得率)',
                  '(吹炼前期耗氧量+吹炼中期耗氧量)/（(TSC检测温度-铁水温度)*(铁水加入量+废钢量)*收得率）',
                  '吹炼后期升温速度',
                  '吹炼前期中期平均升温速度',
                  '供氧强度',
                  '(TSC检测温度-废钢的初始室温)/((TSC时刻-吹炼开始时刻)* 废钢加入量)',
                  '矿石加入量/((TSO检测温度-铁水温度)*(铁水加入量+废钢加入量)*收得率）',
                  'Ca系辅料加入量/((TSO检测温度-铁水温度)*(铁水加入量+废钢加入量)*收得率）',
                  'Mg系辅料加入量/((TSO检测温度-铁水温度)*(铁水加入量+废钢加入量)*收得率）',
                  'TSO温度减目标温度差值']

        column1 = ['铁水加入量', '铁水温度', '铁水Si', '铁水Mn', '废钢量',
                   'TSC检测碳含量', 'TSC检测温度', 'TSC温度减TSO温度',
                   '吹炼前中期_辅料加入总量', 'TSC之后_辅料加入总量',
                   '氧枪高度吹炼前中期均值', '气体压力1吹炼全程均值', '气体压力2吹炼前期均值',
                   '吹炼全程耗氧量/3600']

        df_for = data

        filtered_columns = [col for col in data.columns if col not in column]
        v_model_gbrbm = column1
        DATA_all = pd.concat([Clean_normal1[v_model_gbrbm], Clean_abnormal1[v_model_gbrbm]], axis=0)

        # 保存文件
        # def save_file_to_local_path_sta(file, local_path):
        #     file.to_csv(local_path, encoding='utf-8-sig')
        #     st.success(f"文件已保存到本地路径: {local_path}")

        # 设置初始值字典
        initial_selected_variables = {variable: True for variable in column1}
        # 使用 st.session_state 存储和同步值
        if 'selected_variables' not in st.session_state:
            st.session_state.selected_variables = initial_selected_variables
        if 'selected_variables_z' not in st.session_state:
            st.session_state.selected_variables_z = {variable: True for variable in column}

        # 创建布局
        col1, col2 = st.columns([2, 2])
        col3, col4 = st.columns([2, 2])

        # 在每个列中添加相应的内容
        with col1:
            with st.expander("待选择的原始变量"):
                st.session_state.selected_variables = {
                    variable: st.checkbox(variable, key=f"checkbox_original_{variable}",
                                          value=st.session_state.selected_variables.get(variable, False))
                    for variable in filtered_columns
                }

        with col2:
            st.write("选择参与建模的变量:",
                     [variable for variable, selected in st.session_state.selected_variables.items() if selected])

        with col3:
            with st.expander("待选择的自定义变量"):
                st.session_state.selected_variables_z = {
                    variable: st.checkbox(variable, key=f"checkbox_custom_{variable}",
                                          value=st.session_state.selected_variables_z.get(variable, False))
                    for variable in column
                }

        with col4:
            st.write("选择参与建模的变量:",
                     [variable for variable, selected in st.session_state.selected_variables_z.items() if selected])

        # 合并选择的列
        selected_columns = [variable for variable, selected in st.session_state.selected_variables.items() if
                            selected] + \
                           [variable for variable, selected in st.session_state.selected_variables_z.items() if
                            selected] + \
                            ['温度控制目标符合标签']

        # 输入变量组合
        default_formula = "铁水加入量+废钢量"
        col5, col6 = st.columns([2, 2])

        with col5:
            formula = st.text_input("输入新变量计算公式（e.g., 铁水加入量+废钢量）:")
            st.write(f"计算公式: {formula}")

        with col6:
            st.write("原始列名:", filtered_columns)

        # 检查用户是否输入了公式
        formula1 = []
        data_se = pd.DataFrame()  # Initialize data_se outside the conditional block

        if formula:
            try:
                result_column_df = df_for.apply(lambda row: eval(formula, row.to_dict()), axis=1)
                df_for[formula] = result_column_df
                formula1 = [formula]
                data_se = df_for[selected_columns + formula1]
            except Exception as e:
                st.write(f"发生错误：{e}")

        show_j = st.button('显示结果', key='show_j')

        if show_j:
            if formula1:
                st.dataframe(data_se)
            else:
                st.dataframe(df_for[selected_columns])

        # 保存文件
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
        # localpath1 = st.text_input("变量选择后保存路径（包括文件名.csv）:",
        #                            os.path.join(dirname,
        #                                         '变量选择后文件_{}.csv'.format(formatted_time)
        #                                         )
        #                            )
        #
        # save_data_button_sta = st.button("保存文件", key='bcun')
        # if save_data_button_sta:
        #     if not data_se.empty:
        #         save_file_to_local_path_sta(data_se, localpath1)
        #     else:
        #         save_file_to_local_path_sta(df_for[selected_columns], localpath1)
        if not data_se.empty:
            # save_file_to_local_path_sta(data_se, localpath1)
            download_result(data_se, formatted_time)
        else:
            download_result(df_for[selected_columns], formatted_time)

    else:
        # 如果没有文件上传，显示提示
        st.write("请上传文件")




def show_base3():
    def read_file(uploaded_file):
        # 获取上传文件的名称和扩展名
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1].lower()

        # 将上传的文件保存到临时文件中
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        # 根据扩展名读取文件
        if file_extension == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        elif file_extension == '.csv':
            data = pd.read_csv(file_path)
        elif file_extension in ('.xlsx', '.xls'):
            data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")

        # 删除临时文件
        os.remove(file_path)

        return data

    # 定义阈值函数
    def threshold_func(x):
        if pd.api.types.is_numeric_dtype(x.dtype):  # 如果数据类型是数字类型
            return [x.min(), x.max()]
        else:
            return [np.nan, np.nan]  # 对于非数字类型，设置为空值

    def save_file_to_local_path_sta(file, local_path):
        file.to_csv(local_path, encoding='utf-8-sig')
        st.success(f"文件已保存到本地路径: {local_path}")

    def build_dipin(data1, data2, data3, i):
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.write(data1[i])
        with col2:
            # st.write(data2[i])
            selected_op = st.selectbox(f'原类型：{data2[i]}', options=options_list, key=f'geshi_{i}')
            if selected_op == 'int64':
                result_types.iloc[i, 1] = 'int64'
            elif selected_op == 'float64':
                result_types.iloc[i, 1] = 'float64'
            elif selected_op == 'string':
                result_types.iloc[i, 1] = 'string'
            elif selected_op == 'object':
                result_types.iloc[i, 1] = 'object'

        with col3:
            data3[i] = st.text_input('', key=f'yuzhi_{i}', value=data3[i])
        with col4:
            mean0_key = f'mean_{i}'
            mean0 = st.checkbox('Mean', key=mean0_key, value=True if xu_da_sta1.iloc[0, i] == 1 else False)
            if mean0:
                xu_da_sta1.iloc[0, i] = 1
                mode0_key = f'mode_{i}'
                # 如果 mode0 被选中，则取消勾选，并将 mode0 的状态设置为 False
                if st.session_state.get(mode0_key, False):
                    st.session_state[mode0_key] = False
            else:
                xu_da_sta1.iloc[0, i] = 0  # 如果取消勾选 'Mean' 复选框，则将对应的值设置为0

        with col5:
            mode0_key = f'mode_{i}'
            mode0 = st.checkbox('Mode', key=mode0_key)
            if mode0:
                xu_da_sta1.iloc[0, i] = 2
                mean0_key = f'mean_{i}'
                # 如果 mean0 被选中，则取消勾选
                if st.session_state.get(mean0_key, False):
                    st.session_state[mean0_key] = False

        with col6:
            yichang0 = st.checkbox('Yichang', key=f'yichang_{i}', value=True if xu_da_sta2.iloc[0, i] == 3 else False)
            if yichang0:
                xu_da_sta2.iloc[0, i] = 3

    # 从字符串中提取最小值和最大值
    def extract_min_max(cell):
        try:
            values = ast.literal_eval(cell)  # 使用ast.literal_eval将字符串转换为列表
            values = [x if not np.isnan(x) else None for x in values]  # 将nan替换为None
            return min(values), max(values)
        except (SyntaxError, ValueError):  # 处理无法解析的字符串或NaN
            return np.nan, np.nan

    def convert_to_integer(x):
        try:
            if isinstance(x, str):
                if '.' in x:
                    return int(float(x))  # 将数字字符串转为整数
                elif x.isdigit():
                    return int(x)  # 将纯整数字符串转为整数
                else:
                    return x  # 对于其他字符串，保持原样
            elif pd.notna(pd.to_numeric(x, errors='coerce')):  # 判断是否为浮点数
                return int(x)  # 将浮点数转为整数并四舍五入
        except ValueError:
            pass  # 忽略转换失败的情况，保持原样
        return x  # 其他情况保持原样

    # 生成结果文件并提供下载链接
    def download_result(data1, formatted_time):
        df = data1

        # 将 DataFrame 转换为 CSV 格式的字节流
        csv = df.to_csv(index=False).encode('utf-8-sig')

        # 将字节流转换为 BytesIO 对象
        csv_io = BytesIO(csv)

        # 提供下载链接
        st.download_button(label="下载文件至本地路径", data=csv_io, file_name='低频处理后的炉次_{}.csv'.format(
            formatted_time), mime='text/csv')

    # 低频数据筛选==============================================================
    st.title("转炉质量根因分析模型离线仿真器")
    st.subheader("2-低频数据筛选")

    uploaded_file_sta = st.file_uploader("上传低频数据", type=['pkl', 'csv', 'xlsx'],
                                         key="sta_file_uploader_1")  # ,'csv'
    # 检查是否有文件上传
    if uploaded_file_sta is not None:

        # 1、读取低频数据======================
        file = '必要数据'
        # df_sta = pd.read_pickle(os.path.join(file,'ZI 低频-脱敏发出20230823.pkl'))
        df_sta = read_file(uploaded_file_sta)
        # 读取双渣操作的炉次
        ZI_shuangzha = pd.read_excel(os.path.join(file, '双渣炉次炉次号20230918.xlsx'), sheet_name='ZI')  # 读取低频数据
        # 2、获取低频数据和高频数据的中文表头
        df_sta_name = pd.read_excel(os.path.join(file, '低频数据和高频数据表头20230823.xlsx'),
                                    sheet_name='低频数据')  # 读取低频数据
        df_yuzhi = pd.read_excel(os.path.join(file, '各变量阈值上下限.xlsx'))
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
        st.success("低频数据载入成功,载入{}炉次低频数据".format(df_sta.shape[0]))
        # 3、替换掉脱敏数据中的代码（列名），使用真实中文变量名
        # df_sta.columns = df_sta_name.iloc[0, :]

        # 4、添加计算量
        fl_sec = ['预装', '前装', '吹炼前期', '吹炼中期', 'TSC之后', '后装']
        for i in range(6):
            df_sta['{}_辅料加入总量'.format(fl_sec[i])] = df_sta[
                ['{}_辅原料{}加入量'.format(fl_sec[i], j) for j in range(1, 16)]].sum(
                axis=1)
        v_sta_name = df_sta.columns.tolist()

        data_sta1 = df_sta.copy()
        nan_tso = df_sta[df_sta['温度控制目标符合标签'].isna()].index.tolist()
        row_index = np.unique(nan_tso)
        data_sta1.drop(index=row_index, axis=0, inplace=True)
        if 'TSO检测碳含量' in data_sta1:
            data_sta1["TSO检测碳含量"] = pd.to_numeric(data_sta1["TSO检测碳含量"], errors="coerce")

        ZI_shuangzha_sum = []
        for i in range(ZI_shuangzha.shape[0]):
            ZI_shuangzha_name = data_sta1[data_sta1['标识列'] == ZI_shuangzha.iloc[i][0]].index.tolist()
            ZI_shuangzha_sum.extend(ZI_shuangzha_name)

        shuangzha_remove_da = df_sta.loc[ZI_shuangzha_sum]

        shuangzha_remove_da = shuangzha_remove_da.applymap(convert_to_integer)
        if 'TSO检测碳含量' in shuangzha_remove_da:
            shuangzha_remove_da['TSO检测碳含量'] = shuangzha_remove_da['TSO检测碳含量'].astype(str)
        data_sta1.drop(index=np.array(ZI_shuangzha_sum), axis=0, inplace=True)
        st.success(
            '去除双渣炉次{}，共有{}炉次低频数据含有温度标签'.format(shuangzha_remove_da.shape[0], data_sta1.shape[0]))

        # 获取最小最大值（阈值）
        thresholds = data_sta1.agg(threshold_func)
        # 遍历theresholds的每一列
        for column in thresholds.columns:
            # 如果列名在table2的第一列中
            if column in df_yuzhi.iloc[:, 0].values:
                # 找到table2中对应的第二列的值
                new_value = df_yuzhi[df_yuzhi.iloc[:, 0] == column].iloc[0, 1]
                # 将table1中对应列的第一行修改为新值
                thresholds.at[0, column] = new_value
                # 将table1中对应列的第二行修改为新值
                if not pd.isnull(df_yuzhi[df_yuzhi.iloc[:, 0] == column].iloc[0, 2]):
                    thresholds.at[1, column] = df_yuzhi[df_yuzhi.iloc[:, 0] == column].iloc[0, 2]

        thresholds_list = thresholds.apply(lambda x: f'[{x[0]}, {x[1]}]').tolist()

        cleaned_da = data_sta1[data_sta1['温度控制目标符合标签'] == 0]
        norclea_da = data_sta1[data_sta1['温度控制目标符合标签'] == 1]
        data_types = data_sta1.dtypes
        data_types_0 = []
        for col in data_sta1.columns:
            data_types_0.append({"Column Name": col, "Data Type": data_sta1[col].dtype})

        # 将结果放入 DataFrame
        result_types = pd.DataFrame(data_types_0)

        # 获取温度不命中的均值和众数
        numeric_columns_0 = cleaned_da.select_dtypes(exclude=['object'])
        mean_values_cle = numeric_columns_0.mean()
        mode_values_cle = numeric_columns_0.mode()
        mode_values_cle0 = mode_values_cle.iloc[0]

        # 获取温度命中的均值和众数
        numeric_columns_1 = norclea_da.select_dtypes(exclude=['object'])
        mean_values_nor = numeric_columns_1.mean()
        mode_values_nor = numeric_columns_1.mode()
        mode_values_nor0 = mode_values_nor.iloc[0]

        # 用均值填充
        cleaned_da0 = cleaned_da.fillna(mean_values_cle)
        norclea_da0 = norclea_da.fillna(mean_values_nor)
        data_sta_mean = pd.concat([cleaned_da0, norclea_da0], axis=0)
        if 'TSO检测碳含量' in data_sta_mean:
            data_sta_mean["TSO检测碳含量"] = pd.to_numeric(data_sta_mean["TSO检测碳含量"], errors="coerce")

        # 用众数填充
        cleaned_da1 = cleaned_da.fillna(mode_values_cle0)
        norclea_da1 = norclea_da.fillna(mode_values_nor0)
        data_sta_mode = pd.concat([cleaned_da1, norclea_da1], axis=0)
        if 'TSO检测碳含量' in data_sta_mode:
            data_sta_mode["TSO检测碳含量"] = pd.to_numeric(data_sta_mode["TSO检测碳含量"], errors="coerce")

        columns_da1 = data_sta1.columns
        columns_name_fe = ['铁水加入量', '铁水温度', '铁水C', '铁水Si', '铁水Mn', '铁水P', '铁水S', '废钢量']
        # 创建一个字典，将每个列名映射到值为 1
        values_da = {col: 0 for col in columns_da1}
        # 创建一个新的 DataFrame
        xu_da_sta1 = pd.DataFrame(values_da, index=[0])
        xu_da_sta1.loc[xu_da_sta1.index[0], columns_name_fe] = 1
        xu_da_sta2 = pd.DataFrame(values_da, index=[0])
        xu_da_sta2.loc[xu_da_sta2.index[0], columns_name_fe] = 3

        if 'da_type' not in st.session_state:
            st.session_state.da_type = None
        options_list = ['int64', 'float64', 'string', 'object']

        col_1, col_2, col_3, col_4, col_5, col_6 = st.columns(6)
        with col_1:
            st.write('列名')
        with col_2:
            st.write('数据类型')
        with col_3:
            st.write('阈值')
        with col_4:
            st.write('均值填充')
        with col_5:
            st.write('众数填充')
        with col_6:
            st.write('异常值剔除')

        for i in range(len(data_types)):
            mode_key = f'mode_{i}'
            st.session_state[mode_key] = st.session_state.get(mode_key, False)
            build_dipin(v_sta_name, data_types, thresholds_list, i)

        col_11, col_12, col_13, col_14, col_15 = st.columns(5)
        with col_12:
            ensure_da = st.button('确认')
        with col_14:
            baocun = st.button('保存')

        # 定义结果的会话状态
        if 'result_me' not in st.session_state:
            st.session_state.result_me = None

        if 'result_mo' not in st.session_state:
            st.session_state.result_mo = None

        if 'result_yic' not in st.session_state:
            st.session_state.result_yic = None

        if 'ori' not in st.session_state:
            st.session_state.ori = None

        if 'result_di' not in st.session_state:
            st.session_state.result_di = None

        if 'out_nor' not in st.session_state:
            st.session_state.out_nor = False

        if 'outpath' not in st.session_state:
            st.session_state.outpath = None

        if 'indices_y' not in st.session_state:
            st.session_state.indices_y = None

        result_a = st.empty()
        result_b = st.empty()
        result_c = st.empty()
        result_d = st.empty()

        if ensure_da:
            indices_mo = xu_da_sta1.columns[xu_da_sta1.eq(2).any()]
            indices_me = xu_da_sta1.columns[xu_da_sta1.eq(1).any()]
            indices_yi = xu_da_sta2.columns[xu_da_sta2.eq(3).any()]
            st.session_state.indices_y = indices_yi
            indices_ori = xu_da_sta1.columns[xu_da_sta1.eq(0).any()]
            st.session_state.result_mo = data_sta_mode[indices_mo]
            st.session_state.result_me = data_sta_mean[indices_me]
            st.session_state.ori = data_sta1[indices_ori]
            result_di = pd.concat([data_sta_mean[indices_me], data_sta_mode[indices_mo], data_sta1[indices_ori]],
                                  axis=1)
            # 将列表转换为 DataFrame
            thresholds_list_df = pd.DataFrame(thresholds_list)
            # 获取最小值和最大值
            min_values, max_values = zip(*thresholds_list_df[0].apply(extract_min_max))
            # 创建新的DataFrame
            new_df = pd.DataFrame([min_values, max_values], columns=thresholds_list_df.index)
            new_df.columns = thresholds.columns
            # st.write(xu_da_sta1)
            # st.write(xu_da_sta2)
            # st.dataframe(result_types)
            # 遍历 result_types 表格中的每一行
            for index, row in result_types.iterrows():
                col_name = row["Column Name"]  # 获取列名
                data_type = row["Data Type"]  # 获取数据类型

                # 尝试将 data 表格中对应列的数据类型修改为 result_types 中指定的数据类型
                try:
                    result_di[col_name] = result_di[col_name].astype(data_type)
                except ValueError:
                    # 如果出现异常，保持原来的数据类型
                    pass

            # 清除异常炉次
            st.session_state.result_yic = None
            # 遍历每个列序列
            for ii in indices_yi:
                column_data = result_di.loc[:, ii]
                valid_indices = (column_data >= new_df.at[0, ii]) & (
                        column_data <= new_df.at[1, ii])
                invalid_indices = ~valid_indices  # 取反，获取不在范围内的行索引
                st.session_state.result_yic = pd.concat([st.session_state.result_yic, result_di[invalid_indices]])
                result_di = result_di[valid_indices]

            # 将剩余有效数据保存到 result_di 中
            st.session_state.result_di = result_di

        if baocun:
            if st.session_state.result_me is not None:
                with result_a:
                    expander_mean = st.expander("均值填充结果：")
                    expander_mean.dataframe(st.session_state.result_me)

            if st.session_state.result_mo is not None:
                with result_b:
                    expander_mode = st.expander('众数填充结果：')
                    expander_mode.dataframe(st.session_state.result_mo)

            if st.session_state.result_yic is not None:
                with result_c:
                    expander_yic = st.expander('异常值剔除结果：剔除{}炉次'.format(st.session_state.result_yic.shape[0]))
                    expander_yic.dataframe(st.session_state.result_yic[st.session_state.indices_y])

            if st.session_state.result_di is not None:
                with result_d:
                    expander_di = st.expander("低频筛选后剩余炉次:{}".format(st.session_state.result_di.shape[0]))
                    expander_di.dataframe(st.session_state.result_di)
                    # out_nor = expander_di.button('保存文件', key='out_nor')
                    # localpath = expander_di.text_input("保存路径（包括文件名.csv）:",
                    #                                      os.path.join(file,
                    #                                                   '低频筛选后的炉次_{}.csv'.format(
                    #                                                       formatted_time
                    #                                                   )
                    #                                                   )
                    #                                      )
                    # st.session_state.outpath = localpath
            download_result(st.session_state.result_di, formatted_time)
        # if st.session_state.out_nor:
        #     save_file_to_local_path_sta(st.session_state.result_di, st.session_state.outpath)

    else:
        # 如果没有文件上传，显示提示
        st.write("请上传文件")

def show_base4():
    def del_list(A, B):
        '''
        :param A: 原始列表的索引
        :param B: 要去除的索引
        :return: 新索引
        '''
        new_j = []
        for ii in A:
            if ii not in B:
                new_j.append(ii)
        return new_j

    def group_by_difference(arr, max_difference=50):
        # 对原数组进行排序
        sorted_arr = sorted(arr)
        # 初始化结果列表和当前组列表
        result = []
        current_group = [sorted_arr[0]]
        for i in range(1, len(sorted_arr)):
            # 计算当前数值与上一个数值的差值
            diff = sorted_arr[i] - sorted_arr[i - 1]
            if diff <= max_difference:
                # 将差值小于等于50的数值添加到当前组
                current_group.append(sorted_arr[i])
            else:
                # 差值超过50时，将当前组添加到结果列表，并重新初始化当前组
                result.append(current_group)
                current_group = [sorted_arr[i]]
        # 添加最后一组到结果列表
        result.append(current_group)
        return result

    def continue_lis(a):
        """
        # 有一个列表a=[1,2,3,8,6,7,5,10,16,98,99,100,101] 不考虑数字的顺序
        # 找出连续的数字
        """
        # a = [1,2,3,8,6,7,5,10,16,98,99,100,101,106,107,108]
        result = []
        s = []  # 空栈
        for i in sorted(set(a)):
            if len(s) == 0 or s[-1] + 1 == i:
                s.append(i)  # 入栈
            else:
                if len(s) >= 2:
                    # print(s)
                    result.append(s)
                s = []  # 清空
                s.append(i)  # 入栈
        # 最后一轮，需判断下
        if len(s) >= 2:
            # print(s)
            result.append(s)
        return result

    def get_start_end(ans):
        result = continue_lis(ans)  # 看看ans有多少段，连续的，氧气流量大于0的索引
        relult_len = [len(result[i]) for i in range(len(result))]  # 每一段氧气流量大于0的时间长度
        max_len = max(relult_len)  # 取最长的那一段作为实际吹炼时间段
        index_max = relult_len.index(max_len)
        ans_new = result[index_max]
        return ans_new

    # 找出连续相同数值出现的第一个位置
    def find_yq_start(data, threshold=5):
        count = 1
        first_position = None
        for i in range(1, len(data)):
            if data[i] == data[i - 1]:
                count += 1
            else:
                count = 1
            if count > threshold and first_position is None:
                first_position = i - threshold + 1
        return first_position

    # 找出连续相同数值出现的最后一个位置
    def find_yq_end(data, threshold=5):
        count = 1
        last_position = None
        for i in range(1, len(data)):
            if data[i] == data[i - 1]:
                count += 1
            else:
                count = 1
            if count > threshold:
                last_position = i
        return last_position

    # 删除列表中的元素的函数
    def del_list(original_list, to_delete):
        return [item for item in original_list if item not in to_delete]

    # 文件路径
    file_path = '必要数据'
    pickle_file_path = os.path.join(file_path, 'ZI 低频-脱敏发出20230823.pkl')
    excel_file_path = os.path.join(file_path, '双渣炉次炉次号20230918.xlsx')
    df_sta_name_path = os.path.join(file_path, '低频数据和高频数据表头20230823.xlsx')

    # 读取数据
    df_sta = pd.read_pickle(pickle_file_path)
    ZI_shuangzha = pd.read_excel(excel_file_path, sheet_name='ZI')
    df_sta_name = pd.read_excel(df_sta_name_path, sheet_name='低频数据')
    df_ts_name = pd.read_excel(df_sta_name_path, sheet_name='高频数据')
    df_sta.columns = df_sta_name.iloc[0, :]

    df_ts = pd.read_pickle(os.path.join(file_path, 'ZI 高频-脱敏发出20230823.pkl'))
    df_ts.columns = df_ts_name.iloc[0, :]

    st.title("转炉质量根因分析模型离线仿真器")
    # 创建一个二级标题
    st.subheader("3-高频数据筛选")

    def load_ts_data():
        df_ts = pd.read_pickle(os.path.join(file_path, 'ZI 高频-脱敏发出20230823.pkl'))
        return df_ts

    uploaded_file_sta = st.file_uploader("上传高频数据", type=['xlsx', 'rar', 'zip'],
                                         key="ts_file_uploader_1")  # ,'csv'

    def highlight_all_cells(x, a):
        df_styled = x.copy()
        df_styled = df_styled.style.apply(
            lambda row: [f'color: red' if [row.name, col] in a else '' for col in df_styled.columns], axis=1)
        return df_styled

    modified_rows_dstype = []
    modified_rows_min = []  # 用于存储已修改的行索引或列名
    modified_rows_max = []

    # 检查是否有文件上传
    if uploaded_file_sta is None:
        # 如果没有文件上传，显示提示
        st.write("请上传文件以载入数据")

    else:
        # # 读取上传的低频数据
        # df_ts = load_ts_data()
        # # 3、替换掉脱敏数据中的代码（列名），使用真实中文变量名
        # df_ts.columns = df_ts_name.iloc[0, :]
        # st.write("高频数据载入成功")
        st.success("高频数据载入成功,载入{}炉次高频数据".format(631))
        # st.write(df_ts)

        # kaishi_1, kaishi_2, kaishi_3 = st.columns([1, 1, 1])
        # kaishi_1.write("高频数据筛选前的数据:")
        # kaishi_2.write(631)

        # st.write(pd.DataFrame({"列名": df_ts.columns, "数据类型": df_ts.dtypes}).reset_index())
        DF = pd.DataFrame({"列名": df_ts.columns[2:], "数据类型": list(df_ts.dtypes)[2:]})
        DF['最小值'] = list(df_ts.min())[2:]
        DF['最大值'] = list(df_ts.max())[2:]

        new_data = {'列名': ["TSC时刻", "TSO时刻"], '数据类型': ["int64", "int64"]}
        new_rows = pd.DataFrame(new_data)
        DF = pd.concat([DF, new_rows], ignore_index=True)

        # 显示表格
        editable_df = DF.copy()

        with st.expander("数据类型"):
            for index, row in editable_df.iterrows():
                col_name = row['列名']
                # 检查数据类型是否为NaN，如果是，则将其替换为默认数据类型
                data_types = ['int64', 'float64', 'string']
                if pd.isnull(row['数据类型']):
                    default_data_type = 'int64'
                    editable_df.loc[index, '数据类型'] = default_data_type
                    data_type_index = data_types.index(default_data_type)
                else:
                    data_type_index = data_types.index(row['数据类型'])

                # 使用selectbox函数创建下拉菜单，并将选中的数据类型更新到DataFrame中
                data_type = st.selectbox(f'数据类型 for {col_name}', data_types, index=data_type_index,
                                         key=f'data_type_{index}')
                editable_df.loc[index, '数据类型'] = data_type

                if data_type != str(row['数据类型']):
                    modified_rows_dstype.append([index, '数据类型'])

        with st.expander("阈值"):
            for index, row in editable_df.iterrows():
                col_name = row['列名']
                # 创建两列布局
                col1, col2 = st.columns(2)
                # 在第一列中显示最小值编辑框
                with col1:
                    min_val = st.text_input(f'{col_name}最小值', row['最小值'])
                # 在第二列中显示最大值编辑框
                with col2:
                    max_val = st.text_input(f'{col_name}最大值', row['最大值'])

                    if min_val != str(row['最小值']):
                        modified_rows_min.append([index, '最小值'])
                    if max_val != str(row['最大值']):
                        modified_rows_max.append([index, '最大值'])

                # 更新DataFrame中的值
                editable_df.loc[index, '最小值'] = min_val
                editable_df.loc[index, '最大值'] = max_val

        editable_df['缺失值剔除'] = True
        with st.expander("缺失值剔除"):
            for i in range(len(DF)):
                key = f"checkbox_missing_{i}"
                editable_df.loc[i, '缺失值剔除'] = st.checkbox(f"变量名称: {editable_df.loc[i, '列名']}",
                                                               editable_df.loc[i, '缺失值剔除'], key=key)

        editable_df['异常值处理'] = True
        with st.expander("异常值剔除"):
            for i in range(len(DF)):
                key = f"checkbox_outlier_{i}"
                editable_df.loc[i, '异常值处理'] = st.checkbox(f"变量名称: {editable_df.loc[i, '列名']}",
                                                               editable_df.loc[i, '异常值处理'], key=key)

        # 显示更新后的表格
        # 应用样式到DataFrame
        df_styled = highlight_all_cells(editable_df, modified_rows_min + modified_rows_max + modified_rows_dstype)
        # 将DataFrame转换为HTML表格
        table = df_styled.to_html(escape=False)
        # 在Streamlit中显示表格
        st.write(table, unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        # 添加确认和保存按钮
        queren = col2.button('确认')
        baocun = col4.button('保存')

        # 7.2、增加一些计算量（辅料的6个阶段）
        fl_sec = ['预装', '前装', '吹炼前期', '吹炼中期', 'TSC之后', '后装']
        for i in range(6):
            df_sta['{}_辅料加入总量'.format(fl_sec[i])] = df_sta[
                ['{}_辅原料{}加入量'.format(fl_sec[i], j) for j in range(1, 16)]].sum(axis=1)
        v_sta_name = df_sta.columns.tolist()

        # 7.3、统计低频数据中每行的空值和0值
        null_all = pd.DataFrame(df_sta.isnull().sum(), columns=['空值个数'])  # 统计出每列的空值
        zero_all = pd.DataFrame((df_sta.loc[2:, ] == 0).astype(int).sum(axis=0), columns=['0值个数'])  # 统计出每列的0值
        result = null_all.join(zero_all)
        result['空值和0值个数总和'] = result['空值个数'] + result['0值个数']

        # 7.4、根据空值和0值个数选择参与建模的废钢牌号和辅料类别，空值0值总数越多的变量，则不参与建模
        n_ori = df_sta.shape[0]
        alpha = 0.8  # 这个值越大，保留的辅料变量越多，相当于越宽松，可设置参数1号
        v_name_fg = result['废钢1加入量':'废钢11加入量'][
            result['废钢1加入量':'废钢11加入量']['空值和0值个数总和'] < alpha * n_ori].index.tolist()
        v_name_fu = result['预装_辅原料1加入量':'辅原料15加入总量'][
            result['预装_辅原料1加入量':'辅原料15加入总量']['空值和0值个数总和'] < alpha * n_ori].index.tolist()
        v_name_fus = result['预装_辅料加入总量':'后装_辅料加入总量'][
            result['预装_辅料加入总量':'后装_辅料加入总量']['空值和0值个数总和'] < alpha * n_ori].index.tolist()
        v_name_fix = ['标识列', '班组', '铁水前S', '铁水加入量', '铁水温度', '铁水Si', '铁水Mn', '铁水P', '铁水S',
                      '铁水比',
                      '废钢量', 'TSC检测碳含量', 'TSC检测温度', 'TSO检测温度', '目标温度']
        v_name_y = ['温度控制目标符合标签']
        # 确定参与建模的变量
        v_name_all = v_name_fix + v_name_fg + v_name_fu + v_name_fus + v_name_y

        # 可得到v_name_fus = ['吹炼前期_辅料加入总量', '吹炼中期_辅料加入总量', 'TSC之后_辅料加入总量', '后装_辅料加入总量']
        # 有的炉次的'吹炼前期_辅料加入总量'为0，预装和前装不为0，因此，将'吹炼前期_辅料加入总量'用 【预装、前装和吹炼前期的总和】 代替
        df_sta['吹炼前期_辅料加入总量_new'] = df_sta['预装_辅料加入总量'] + \
                                              df_sta['前装_辅料加入总量'] + \
                                              df_sta['吹炼前期_辅料加入总量']

        v_name_all[v_name_all.index('吹炼前期_辅料加入总量')] = '吹炼前期_辅料加入总量_new'

        # 替换列名
        v_name_empty = ['标识列', '班组', '铁水前S', '铁水加入量', '铁水温度', '铁水Si', '铁水Mn', '铁水P', '铁水S',
                        '铁水比',
                        '废钢量', 'TSC检测碳含量', 'TSC检测温度']
        data_sta0 = df_sta[v_name_empty].replace(0, np.nan)
        v_empty_other = del_list(v_name_all, v_name_empty)
        data_sta1 = data_sta0.join(df_sta[v_empty_other])  # 跟其他列合并

        # 去除没有温度命中标签的炉次
        nan_tso = data_sta1[data_sta1['温度控制目标符合标签'].isna()].index.tolist()
        # 删除指定的一行
        row_index = np.unique(nan_tso)
        data_sta1.drop(index=row_index, axis=0, inplace=True)

        ZI_shuangzha_sum = []
        for i in range(ZI_shuangzha.shape[0]):
            ZI_shuangzha_name = data_sta1[data_sta1['标识列'] == ZI_shuangzha.iloc[i][0]].index.tolist()
            ZI_shuangzha_sum.extend(ZI_shuangzha_name)
        data_sta1.drop(index=np.array(ZI_shuangzha_sum), axis=0, inplace=True)

        # if st.button('双渣法炉次剔除'):
        #     st.text("双渣法炉次剔除: {}".format(data_sta1.shape[0]))

        Clean_1 = data_sta1
        # 使用均值替代 NaN 值
        # 命中炉次的用命中的均值替代，不命中的用不命中的替代
        Clean_1_filled0 = Clean_1[Clean_1['温度控制目标符合标签'] == 0].fillna(
            Clean_1[Clean_1['温度控制目标符合标签'] == 0].mean())
        Clean_1_filled1 = Clean_1[Clean_1['温度控制目标符合标签'] == 1].fillna(
            Clean_1[Clean_1['温度控制目标符合标签'] == 1].mean())
        Clean_1_filled = pd.concat((Clean_1_filled0, Clean_1_filled1))
        data_model_sta = Clean_1_filled[v_name_all]  #
        data_model_sta['TSC温度减TSO温度'] = data_model_sta['TSC检测温度'] - data_model_sta['TSO检测温度']
        name_all_other = del_list(df_sta.columns.tolist(), data_model_sta.columns.tolist()) + ['标识列']
        data_model_sta_all = pd.merge(data_model_sta, df_sta[name_all_other], on='标识列')
        # 空值处理
        # if st.button('均值填充'):
        #     # 显示均值填充后的数据
        #     st.write("均值填充后的数据:")
        #     st.write(data_model_sta_all)

        df_ts = pd.read_pickle(os.path.join(file_path, 'ZI 高频-脱敏发出20230823.pkl'))
        # 3、替换掉脱敏数据中的代码（列名），使用真实中文变量名
        df_ts.columns = df_ts_name.iloc[0, :]
        # 4、处理高频数据中的时间列，将其转换为与低频数据时刻对应的时间格式
        df_ts['时刻'] = df_ts['时间列'].map(lambda x: time.strftime("%Y%m%d%H%M%S", time.localtime(round(x / 1000))),
                                            na_action='ignore')
        # 5、找出低频数据和高频数据重复的炉次号
        number = df_ts['标识列'].unique()  # 得到1052个炉次的熔炼号
        number = pd.DataFrame(number, columns=['标识列'])
        data_sta = pd.merge(number, data_model_sta_all, on='标识列')  # 低频数据和高频炉次号合并

        # st.text("低频数据和高频数据炉次号能对应上的有: {}".format(data_sta.shape[0]))

        # 6.1 先把每个炉次的时间顺序调过来
        Data_ts_len = []  # # 获取每一炉次的采样点个数
        Data_ts = []  # 获取每一炉次的数据
        for i in range(data_sta.shape[0]):
            every_data = df_ts[df_ts['标识列'] == data_sta.loc[i, '标识列']].reset_index()
            every_data = every_data.sort_values(by='index', ascending=False)  # 按照时间逆序排序
            every_data['index'] = np.arange(0, every_data.shape[0], 1)  # 更改index列
            every_data = every_data.reset_index(drop=True)  # 重置索引并丢弃旧索引
            Data_ts_len.append(every_data.shape[0])  # 每一炉次的原始吹炼时间长度
            Data_ts.append(every_data.apply(pd.to_numeric, errors='raise'))
        v_name_ts = ['index'] + df_ts.columns.tolist()  # 增加一个列名

        # 8.2、确定吹炼开始时刻（根据氧枪高度的下枪和提枪截取，更新吹炼开始时刻，TSC时刻和TSO时刻可以不变）
        n_tso_miss = 0
        ych_lu_all = []
        ych_lu_1 = []
        ych_lu_2 = []
        ych_lu_3 = []
        ych_lu_4 = []
        ych_lu_5 = []
        ych_lu_6 = []
        lu_all = []

        ych_1 = []
        ych_2 = []
        ych_3 = []
        ych_4 = []
        ych_5 = []
        ych_6 = []

        # for i in range(len(Data_ts)):
        #     every_data = Data_ts[i]
        #     data_sta2 = data_model_sta_all[data_model_sta_all['标识列']==every_data['标识列'].iloc[0]] # 定位找到标识列对应的低频数据
        #     dc_du = []
        #     for j in ['底吹气体流量1', '底吹气体流量2', '底吹气体流量3', '底吹气体流量4', '底吹气体流量5', '底吹气体流量6', '底吹气体流量7', '底吹气体流量8',
        #                 '底吹气体流量9', '底吹气体流量10', '底吹气体流量11', '底吹气体流量12', '底吹气体流量13', '底吹气体流量14', '底吹气体流量15', '底吹气体流量16']:
        #         if (every_data[j] == 0).all():
        #             dc_du.append(j)
        #     data_sta2['底吹气体流量为0标记'] = str(dc_du)
        #     data_sta2['底吹气体流量为0个数'] = len(dc_du)/2
        #     # 找到对应的吹炼起止时刻
        #     if pd.isnull(data_sta2['吹炼开始时刻']).all():
        #         print('第{}个炉次号为{}的低频数据中缺少吹炼开始时刻'.format(i,every_data['标识列'].iloc[0]))
        #         ych.append(i)
        #         n_tso_miss += 1
        #         ych_lu_1.append(every_data['标识列'].iloc[0])

        # for i in range(len(Data_ts)):
        #     every_data = Data_ts[i]
        #     data_sta2 = data_model_sta_all[data_model_sta_all['标识列'] == every_data['标识列'].iloc[0]]  # 定位找到标识列对应的低频数据
        #     dc_du = []
        #     for j in ['底吹气体流量1', '底吹气体流量2', '底吹气体流量3', '底吹气体流量4', '底吹气体流量5', '底吹气体流量6',
        #               '底吹气体流量7', '底吹气体流量8',
        #               '底吹气体流量9', '底吹气体流量10', '底吹气体流量11', '底吹气体流量12', '底吹气体流量13',
        #               '底吹气体流量14', '底吹气体流量15', '底吹气体流量16']:
        #         if (every_data[j] == 0).all():
        #             dc_du.append(j)
        #     data_sta2['底吹气体流量为0标记'] = str(dc_du)
        #     data_sta2['底吹气体流量为0个数'] = len(dc_du) / 2
        #     if pd.isnull(data_sta2['TSO时刻']).all():
        #         print('第{}个炉次号为{}的低频数据中缺少TSO时刻'.format(i,every_data['标识列'].iloc[0]))
        #         ych.append(i)
        #         n_tso_miss += 1
        #         ych_lu_2.append(every_data['标识列'].iloc[0])

        def queshi_all():
            qsh_all = []
            qsh_all_index = []
            qsh_all_lu = []
            qsh_all_lu_fen = []
            for j in (list(df_ts.columns[2:])):
                if editable_df.loc[editable_df['列名'] == j, '缺失值剔除'].any():
                    qsh_all.append(j)
                    for i in range(len(Data_ts)):
                        every_data = Data_ts[i]
                        null_percentages = every_data[j].isnull().mean()
                        if (null_percentages >= 0.9):
                            print('第{}个炉次{}的{}缺失'.format(i, j, every_data['标识列'].iloc[0]))
                            qsh_all.append(every_data['标识列'].iloc[0])
                            qsh_all_index.append(i)
                            qsh_all_lu.append(every_data['标识列'].iloc[0])
                qsh_all_lu_fen.append(qsh_all)
                qsh_all = []
            return qsh_all_lu_fen, qsh_all_lu, qsh_all_index

        def yichang_all():
            ych_all = []
            ych_all_index = []
            ych_all_lu = []
            ych_all_lu_fen = []
            for j in (list(df_ts.columns[2:])):
                if editable_df.loc[editable_df['列名'] == j, '异常值处理'].any():
                    ych_all.append(j)
                    for i in range(len(Data_ts)):
                        every_data = Data_ts[i]
                        # ans = every_data.index[
                        #     (every_data[j] >= 1046) &
                        #     (every_data[j] <= 1450)].tolist()

                        ans = every_data.index[
                            (every_data[j] >= float(editable_df.loc[editable_df['列名'] == j, '最小值'].values[0])) &
                            (every_data[j] <= float(
                                editable_df.loc[editable_df['列名'] == j, '最大值'].values[0]))].tolist()
                        if len(ans) <= 1:  # 炉次没有氧枪高度小于1450的
                            print('第{}个炉次{}的{}异常'.format(i, j, every_data['标识列'].iloc[0]))
                            ych_all.append(every_data['标识列'].iloc[0])
                            ych_all_index.append(i)
                            ych_all_lu.append(every_data['标识列'].iloc[0])
                ych_all_lu_fen.append(ych_all)
                ych_all = []
            return ych_all_lu_fen, ych_all_lu, ych_all_index

        def TSO_queshi():
            for i in range(len(Data_ts)):
                every_data = Data_ts[i]
                data_sta2 = data_model_sta_all[
                    data_model_sta_all['标识列'] == every_data['标识列'].iloc[0]]  # 定位找到标识列对应的低频数据
                dc_du = []
                for j in ['底吹气体流量1', '底吹气体流量2', '底吹气体流量3', '底吹气体流量4', '底吹气体流量5',
                          '底吹气体流量6',
                          '底吹气体流量7', '底吹气体流量8',
                          '底吹气体流量9', '底吹气体流量10', '底吹气体流量11', '底吹气体流量12', '底吹气体流量13',
                          '底吹气体流量14', '底吹气体流量15', '底吹气体流量16']:
                    if (every_data[j] == 0).all():
                        dc_du.append(j)
                data_sta2['底吹气体流量为0标记'] = str(dc_du)
                data_sta2['底吹气体流量为0个数'] = len(dc_du) / 2
                if every_data[(every_data.时刻 == data_sta2['TSO时刻'].iloc[0])].shape[0] == 0:
                    print('{}低频数据的TSO时刻在高频数据中没有'.format(every_data['标识列'].iloc[0]))
                    ych_3.append(i)
                    ych_lu_3.append(every_data['标识列'].iloc[0])
            return ych_lu_3, ych_3

        # def yangqiang_yichang():
        #     for i in range(len(Data_ts)):
        #         every_data = Data_ts[i]
        #         data_sta2 = data_model_sta_all[data_model_sta_all['标识列'] == every_data['标识列'].iloc[0]]  # 定位找到标识列对应的低频数据
        #         dc_du = []
        #         for j in ['底吹气体流量1', '底吹气体流量2', '底吹气体流量3', '底吹气体流量4', '底吹气体流量5', '底吹气体流量6',
        #                   '底吹气体流量7', '底吹气体流量8',
        #                   '底吹气体流量9', '底吹气体流量10', '底吹气体流量11', '底吹气体流量12', '底吹气体流量13',
        #                   '底吹气体流量14', '底吹气体流量15', '底吹气体流量16']:
        #             if (every_data[j] == 0).all():
        #                 dc_du.append(j)
        #         data_sta2['底吹气体流量为0标记'] = str(dc_du)
        #         data_sta2['底吹气体流量为0个数'] = len(dc_du) / 2
        #         ans = every_data.index[every_data['氧枪高度'] < 1450].tolist()  # 获取氧枪高度小于1450的索引
        #         if len(ans) <= 1:  # 炉次没有氧枪高度小于1450的
        #             print('第{}个炉次{}的氧枪高度没有小于1450的'.format(i,every_data['标识列'].iloc[0]))
        #             ych_4.append(i)
        #             ych_lu_4.append(every_data['标识列'].iloc[0])
        #     return ych_lu_4,ych_4

        def yangqiang_queshi():
            for i in range(len(Data_ts)):
                every_data = Data_ts[i]
                data_sta2 = data_model_sta_all[
                    data_model_sta_all['标识列'] == every_data['标识列'].iloc[0]]  # 定位找到标识列对应的低频数据
                dc_du = []
                for j in ['底吹气体流量1', '底吹气体流量2', '底吹气体流量3', '底吹气体流量4', '底吹气体流量5',
                          '底吹气体流量6',
                          '底吹气体流量7', '底吹气体流量8',
                          '底吹气体流量9', '底吹气体流量10', '底吹气体流量11', '底吹气体流量12', '底吹气体流量13',
                          '底吹气体流量14', '底吹气体流量15', '底吹气体流量16']:
                    if (every_data[j] == 0).all():
                        dc_du.append(j)
                data_sta2['底吹气体流量为0标记'] = str(dc_du)
                data_sta2['底吹气体流量为0个数'] = len(dc_du) / 2
                ans = every_data.index[every_data['氧枪高度'] < 1450].tolist()
                if ans:
                    ans_new = get_start_end(ans)  # # 取最长的那一段作为实际氧枪高度段
                if ans_new[0] == 0:
                    print('第{}个炉次{}的氧枪高度数据不全'.format(i, every_data['标识列'].iloc[0]))
                    ych_5.append(i)
                    ych_lu_5.append(every_data['标识列'].iloc[0])
            return ych_lu_5, ych_5

        def TSO_yichang():
            for i in range(len(Data_ts)):
                every_data = Data_ts[i]
                data_sta2 = data_model_sta_all[
                    data_model_sta_all['标识列'] == every_data['标识列'].iloc[0]]  # 定位找到标识列对应的低频数据
                dc_du = []
                for j in ['底吹气体流量1', '底吹气体流量2', '底吹气体流量3', '底吹气体流量4', '底吹气体流量5',
                          '底吹气体流量6',
                          '底吹气体流量7', '底吹气体流量8',
                          '底吹气体流量9', '底吹气体流量10', '底吹气体流量11', '底吹气体流量12', '底吹气体流量13',
                          '底吹气体流量14', '底吹气体流量15', '底吹气体流量16']:
                    if (every_data[j] == 0).all():
                        dc_du.append(j)
                data_sta2['底吹气体流量为0标记'] = str(dc_du)
                data_sta2['底吹气体流量为0个数'] = len(dc_du) / 2
                ans = every_data.index[every_data['氧枪高度'] < 1450].tolist()
                if ans:
                    ans_new = get_start_end(ans)  # # 取最长的那一段作为实际氧枪高度段
                    repeated_position = find_yq_start(every_data.氧枪高度.iloc[ans_new[0]:
                                                                               ans_new[0] + 100].tolist(), threshold=5)
                    last_position = find_yq_end(every_data.氧枪高度.iloc[ans_new[-1] - 50:ans_new[-1]].tolist(),
                                                threshold=5)
                    yq_start = ans_new[0] + repeated_position if repeated_position is not None else None
                    yq_end = ans_new[-1] - 50 + last_position if last_position is not None else None

                    data_sta2['吹炼开始时刻_new'] = every_data.时刻.iloc[yq_start]  # 以规则得到的为准
                    data_sta2['氧枪提枪时刻'] = every_data.时刻.iloc[yq_end]  # 以规则得到的为准
                    end_tso = every_data[(every_data.时刻 == data_sta2['TSO时刻'].iloc[0])].index.tolist()[0]
                    if yq_start > end_tso:  # 如果吹炼开始时刻在TSO时刻之后
                        print('{}的吹炼开始时刻和TSO时刻顺序不对'.format(every_data['标识列'].iloc[0]))
                        ych_6.append(i)
                        ych_lu_6.append(every_data['标识列'].iloc[0])
            return ych_lu_6, ych_6

        ych_all, ych_all_lu, ych_all_index = yichang_all()
        qsh_all, qsh_all_lu, qsh_all_index = queshi_all()

        LU_1, LU_1_index = yangqiang_queshi()
        TSO_1, TSO_1_index = TSO_queshi()
        TSO_2, TSO_2_index = TSO_yichang()

        Data_sta_2 = []  # 获取新的吹炼开始时刻
        ych_lu_all = set(ych_all_lu + qsh_all_lu + LU_1 + TSO_1 + TSO_2)
        lu_all = list(set(data_sta['标识列']) - ych_lu_all)
        for i in range(len(Data_ts)):
            every_data = Data_ts[i]
            if (every_data['标识列'].iloc[0] in lu_all):
                data_sta2 = data_model_sta_all[data_model_sta_all['标识列'] == every_data['标识列'].iloc[0]]
                ans = every_data.index[every_data['氧枪高度'] < 1450].tolist()
                ans_new = get_start_end(ans)  # # 取最长的那一段作为实际氧枪高度段
                repeated_position = find_yq_start(every_data.氧枪高度.iloc[ans_new[0]:
                                                                           ans_new[0] + 100].tolist(), threshold=5)
                last_position = find_yq_end(every_data.氧枪高度.iloc[ans_new[-1] - 50:ans_new[-1]].tolist(),
                                            threshold=5)
                yq_start = ans_new[0] + repeated_position if repeated_position is not None else None
                yq_end = ans_new[-1] - 50 + last_position if last_position is not None else None

                data_sta2['吹炼开始时刻_new'] = every_data.时刻.iloc[yq_start]  # 以规则得到的为准
                data_sta2['氧枪提枪时刻'] = every_data.时刻.iloc[yq_end]  # 以规则得到的为准
                Data_sta_2.append(data_sta2)
        Data_sta2 = pd.DataFrame(np.array(Data_sta_2).squeeze(), columns=data_sta2.columns)

        # 删除掉吹炼时长异常的炉次
        # 时间格式
        time_format = "%Y%m%d%H%M%S"
        Data_sta2['提枪时刻-吹炼开始时刻'] = (pd.to_datetime(Data_sta2['氧枪提枪时刻'], format=time_format) -
                                              pd.to_datetime(Data_sta2['吹炼开始时刻_new'],
                                                             format=time_format)).dt.total_seconds()
        abnormal_chl_time = []
        normal_chl_time = []
        abnormal_bsl = []
        for i in range(Data_sta2.shape[0]):
            if (Data_sta2['提枪时刻-吹炼开始时刻'].iloc[i] < 720):
                abnormal_chl_time.extend(data_sta[data_sta['标识列'] == Data_sta2['标识列'].iloc[i]].index.tolist())
                abnormal_bsl.append(Data_sta2['标识列'].iloc[i])
            else:
                normal_chl_time.append(Data_sta2['标识列'].iloc[i])

        # 8.3、获取TSC时刻
        # 根据吹炼时刻异常的数据去除的炉次，在高频数据里也去除掉
        ych = list(set(ych_all_index + qsh_all_index + LU_1_index + TSO_1_index + TSO_2_index))
        index_hf0 = del_list(list(range(len(Data_ts))), ych)
        index_hf1 = del_list(index_hf0, abnormal_chl_time)
        print('炉次数{}中{}个炉次吹炼时长小于12分钟，剔除后剩余{}炉次'.format(len(Data_ts) - n_tso_miss,
                                                                             len(abnormal_chl_time), len(index_hf1)))
        # 炉次号
        lu_all_1 = []
        for i in index_hf1:  # len(Data_ts_real)
            every_data = Data_ts[i]
            lu_all_1.append(every_data['标识列'].iloc[0])

        Data_ts_len_real = []  # # 根据吹炼开始时刻和TSO时刻获取每一炉次的实际吹炼采样点个数
        Data_ts_real = []  # 获取每一炉次的数据
        get_tsc = []  # 获得每个炉次的索引及TSC时刻点
        get_tsc_time = []  # 获得每个炉次的TSC时刻点
        get_bsl = []  # 获得每个炉次的炉次号
        abnor = 0  # 计数，筛选掉多少个炉次
        get_start = []  # 获得每个炉次的吹炼开始时刻
        get_tq = []  # 获得每个炉次的索引及 氧枪提枪时刻
        get_tq_time = []  # 获得每个炉次的氧枪提枪时刻

        abnor_all = []
        abnor_1 = []
        abnor_2 = []
        abnor_3 = []
        abnor_4 = []
        abnor_5 = []

        # for i in index_hf1:  #len(Data_ts_real)
        #     every_data = Data_ts[i]
        #     data_sta2 = Data_sta2[Data_sta2['标识列']==every_data['标识列'].iloc[0]] # 定位找到标识列对应的低频数据
        #     end_tso = every_data[(every_data.时刻 == data_sta2['TSO时刻'].iloc[0])].index.tolist()[0]  # 定位找到TSO时刻
        #     start = every_data[(every_data.时刻 == data_sta2['吹炼开始时刻_new'].iloc[0])].index.tolist()[0]
        #     end_cl = every_data[(every_data.时刻 == data_sta2['吹炼终了时刻'].iloc[0])].index.tolist()[0]
        #     every_data_new1 = every_data.loc[start:end_tso, :]  # end_cl
        #     # 找到TSC对应的时刻，利用【副枪信号曲线】去寻找
        #     if every_data[(every_data.时刻 == data_sta2['吹炼终了时刻'].iloc[0])].shape[0] == 0:
        #         print('第{}个炉次号{}低频数据的吹炼终了时刻在高频数据中没有'.format(i,every_data_new1['标识列'].iloc[0]))
        #         abnor += 1
        #         abnor_1.append(every_data_new1['标识列'].iloc[0])

        # for i in index_hf1:  # len(Data_ts_real)
        #     every_data = Data_ts[i]
        #     data_sta2 = Data_sta2[Data_sta2['标识列'] == every_data['标识列'].iloc[0]]  # 定位找到标识列对应的低频数据
        #     end_tso = every_data[(every_data.时刻 == data_sta2['TSO时刻'].iloc[0])].index.tolist()[0]  # 定位找到TSO时刻
        #
        #     start = every_data[(every_data.时刻 == data_sta2['吹炼开始时刻_new'].iloc[0])].index.tolist()[0]
        #     end_cl = every_data[(every_data.时刻 == data_sta2['吹炼终了时刻'].iloc[0])].index.tolist()[0]
        #     every_data_new1 = every_data.loc[start:end_tso, :]  # end_cl
        #     ans = every_data_new1.index[every_data_new1.副枪信号曲线 > 1050].tolist()  # 获取副枪信号曲线大于0的索引
        #     if len(ans)==0:
        #         print('第{}个炉次号{}在吹炼开始到吹炼终了的副枪信号曲线均小于750'.format(i,every_data_new1['标识列'].iloc[0]))
        #         abnor += 1
        #         abnor_2.append(every_data_new1['标识列'].iloc[0])

        # for i in index_hf1:  # len(Data_ts_real)
        #     every_data = Data_ts[i]
        #     data_sta2 = Data_sta2[Data_sta2['标识列'] == every_data['标识列'].iloc[0]]  # 定位找到标识列对应的低频数据
        #     end_tso = every_data[(every_data.时刻 == data_sta2['TSO时刻'].iloc[0])].index.tolist()[0]  # 定位找到TSO时刻
        #
        #     start = every_data[(every_data.时刻 == data_sta2['吹炼开始时刻_new'].iloc[0])].index.tolist()[0]
        #     end_cl = every_data[(every_data.时刻 == data_sta2['吹炼终了时刻'].iloc[0])].index.tolist()[0]
        #     every_data_new1 = every_data.loc[start:end_tso, :]  # end_cl
        #     ans = every_data_new1.index[every_data_new1.副枪信号曲线 > 1050].tolist()  # 获取副枪信号曲线大于0的索引
        #
        #     # 分组，差值小于50的数值分到一组中
        #     groups = group_by_difference(ans, max_difference=50) # 可设置参数3号
        #     tsc_start,tsc_end = groups[0][0], groups[0][-1]
        #     # if (every_data.氧气流量.loc[tsc_start:tsc_end].max()-every_data.氧气流量.loc[tsc_start:tsc_end].min())<5000:
        #     #     print('第{}个炉次号{}没有下副枪'.format(i,every_data_new1['标识列'].iloc[0]))
        #     #     abnor += 1
        #     #     continue
        #     tsc_time_index = every_data.loc[tsc_start:tsc_end][every_data.氧气流量.loc[tsc_start:tsc_end] == every_data.氧气流量.loc[tsc_start:tsc_end].min()].index.tolist()[0]
        #     if end_tso < tsc_time_index:
        #         print('{}的TSC时刻和TSO时刻顺序不对'.format(every_data['标识列'].iloc[0]))
        #         abnor += 1
        #         abnor_3.append(every_data_new1['标识列'].iloc[0])

        def TSO_yichang_1():
            abnor_4 = []
            for i in index_hf1:  # len(Data_ts_real)
                every_data = Data_ts[i]
                data_sta2 = Data_sta2[Data_sta2['标识列'] == every_data['标识列'].iloc[0]]  # 定位找到标识列对应的低频数据
                end_tso = every_data[(every_data.时刻 == data_sta2['TSO时刻'].iloc[0])].index.tolist()[0]  # 定位找到TSO时刻

                start = every_data[(every_data.时刻 == data_sta2['吹炼开始时刻_new'].iloc[0])].index.tolist()[0]
                end_cl = every_data[(every_data.时刻 == data_sta2['吹炼终了时刻'].iloc[0])].index.tolist()[0]
                every_data_new1 = every_data.loc[start:end_tso, :]  # end_cl
                ans = every_data_new1.index[every_data_new1.副枪信号曲线 > 1050].tolist()  # 获取副枪信号曲线大于0的索引

                # 分组，差值小于50的数值分到一组中
                groups = group_by_difference(ans, max_difference=50)  # 可设置参数3号
                tsc_start, tsc_end = groups[0][0], groups[0][-1]
                # 有的炉次的TSO时刻和副枪信号曲线对不上，需要把这些炉次剔除掉
                # TSO时刻在TSC时刻之后
                every_data_new2 = every_data.loc[tsc_end:, :]
                ans_tso = every_data_new2.index[every_data_new2.副枪信号曲线 > 1050].tolist()  # 获取副枪信号曲线大于0的索引
                # 分组，差值小于50的数值分到一组中
                groups_tso = group_by_difference(ans_tso, max_difference=50)  # 可设置参数3号
                tso_start, tso_end = groups_tso[-1][0], groups_tso[-1][-1]
                if (end_tso < tso_start - 10) or (end_tso > tso_end + 10):
                    print('第{}个炉次号{}的TSO时刻不在副枪信号曲线范围内'.format(i, every_data['标识列'].iloc[0]))
                    abnor_4.append(every_data_new1['标识列'].iloc[0])
            return abnor_4

        def TSC_yichang():
            for i in index_hf1:  # len(Data_ts_real)
                every_data = Data_ts[i]
                data_sta2 = Data_sta2[Data_sta2['标识列'] == every_data['标识列'].iloc[0]]  # 定位找到标识列对应的低频数据
                end_tso = every_data[(every_data.时刻 == data_sta2['TSO时刻'].iloc[0])].index.tolist()[0]  # 定位找到TSO时刻

                start = every_data[(every_data.时刻 == data_sta2['吹炼开始时刻_new'].iloc[0])].index.tolist()[0]
                end_cl = every_data[(every_data.时刻 == data_sta2['吹炼终了时刻'].iloc[0])].index.tolist()[0]
                every_data_new1 = every_data.loc[start:end_tso, :]  # end_cl
                ans = every_data_new1.index[every_data_new1.副枪信号曲线 > 1050].tolist()  # 获取副枪信号曲线大于0的索引

                # 分组，差值小于50的数值分到一组中
                groups = group_by_difference(ans, max_difference=50)  # 可设置参数3号
                tsc_start, tsc_end = groups[0][0], groups[0][-1]
                # 有的炉次的TSO时刻和副枪信号曲线对不上，需要把这些炉次剔除掉
                # TSO时刻在TSC时刻之后
                every_data_new2 = every_data.loc[tsc_end:, :]
                ans_tso = every_data_new2.index[every_data_new2.副枪信号曲线 > 1050].tolist()  # 获取副枪信号曲线大于0的索引
                # 分组，差值小于50的数值分到一组中
                groups_tso = group_by_difference(ans_tso, max_difference=50)  # 可设置参数3号
                tso_start, tso_end = groups_tso[-1][0], groups_tso[-1][-1]

                every_data = every_data.reset_index(drop=True)
                tsc_time = every_data.loc[tsc_start:tsc_end][
                    every_data.氧气流量.loc[tsc_start:tsc_end] == every_data.氧气流量.loc[tsc_start:tsc_end].min()].时刻
                tq_time = every_data.loc[every_data.时刻 == data_sta2['氧枪提枪时刻'].iloc[0]].时刻
                if tsc_time.index[0] > tq_time.index[0]:
                    print(
                        '第{}个炉次号{}的TSC在提枪时刻之后，证明数据有问题'.format(i, every_data_new1['标识列'].iloc[0]))
                    abnor_5.append(every_data_new1['标识列'].iloc[0])
            return abnor_5

        TSO_3 = TSO_yichang_1()
        TSC_1 = TSC_yichang()

        # abnor_all = set(abnor_1 + abnor_2 + abnor_3 + abnor_4 + abnor_5)
        abnor_all = set(TSO_3 + TSC_1)
        lu_all_2 = list(set(lu_all_1) - abnor_all)

        for i in index_hf1:  # len(Data_ts_real)
            every_data = Data_ts[i]
            if (every_data['标识列'].iloc[0] in lu_all_2):
                data_sta2 = Data_sta2[Data_sta2['标识列'] == every_data['标识列'].iloc[0]]  # 定位找到标识列对应的低频数据
                end_tso = every_data[(every_data.时刻 == data_sta2['TSO时刻'].iloc[0])].index.tolist()[0]  # 定位找到TSO时刻
                start = every_data[(every_data.时刻 == data_sta2['吹炼开始时刻_new'].iloc[0])].index.tolist()[0]
                end_cl = every_data[(every_data.时刻 == data_sta2['吹炼终了时刻'].iloc[0])].index.tolist()[0]
                every_data_new1 = every_data.loc[start:end_tso, :]  # end_cl
                ans = every_data_new1.index[every_data_new1.副枪信号曲线 > 1050].tolist()  # 获取副枪信号曲线大于0的索引

                # 分组，差值小于50的数值分到一组中
                groups = group_by_difference(ans, max_difference=50)  # 可设置参数3号
                tsc_start, tsc_end = groups[0][0], groups[0][-1]
                # 有的炉次的TSO时刻和副枪信号曲线对不上，需要把这些炉次剔除掉
                # TSO时刻在TSC时刻之后
                every_data_new2 = every_data.loc[tsc_end:, :]
                ans_tso = every_data_new2.index[every_data_new2.副枪信号曲线 > 1050].tolist()  # 获取副枪信号曲线大于0的索引
                # 分组，差值小于50的数值分到一组中
                groups_tso = group_by_difference(ans_tso, max_difference=50)  # 可设置参数3号
                tso_start, tso_end = groups_tso[-1][0], groups_tso[-1][-1]

                every_data = every_data.reset_index(drop=True)
                tsc_time = every_data.loc[tsc_start:tsc_end][
                    every_data.氧气流量.loc[tsc_start:tsc_end] == every_data.氧气流量.loc[tsc_start:tsc_end].min()].时刻
                tq_time = every_data.loc[every_data.时刻 == data_sta2['氧枪提枪时刻'].iloc[0]].时刻

                get_tq.append(tq_time)
                get_tq_time.append(tq_time.iloc[0])
                get_tsc.append(tsc_time)
                get_tsc_time.append(tsc_time.iloc[0])
                get_bsl.append(every_data['标识列'].iloc[0])
                every_data_new = every_data.loc[start:end_tso, :]
                every_data_new = every_data_new.reset_index(drop=True)  # 重置索引并丢弃旧索引
                Data_ts_len_real.append(every_data_new.shape[0])  # 每一炉次的原始吹炼时间长度
                Data_ts_real.append(every_data_new.apply(pd.to_numeric, errors='raise'))
                get_start.append(start)

        print('在获取TSC时刻阶段，发现了{}个异常炉次数据，需要剔除掉'.format(abnor))

        # Streamlit 应用程序
        # 开始筛选按钮
        if 'selected' not in st.session_state:
            st.session_state.selected = False
        if 'selected_1' not in st.session_state:
            st.session_state.selected_1 = False
        if 'selected_2' not in st.session_state:
            st.session_state.selected_2 = False

        # if queshi_1 or queshi_2 or yichang_1 or shike_1 or shike_2:
        if queren or baocun:
            st.session_state.selected = True

        # with st.expander("缺失值剔除"):
        #     if st.session_state.selected:
        #         selected_data_1 = st.multiselect('缺失值剔除：氧枪高度({}个)'.format(len(LU_1)), options=LU_1)
        #         if selected_data_1:
        #             for i in range(len(Data_ts)):
        #                 every_data = Data_ts[i]
        #                 if (every_data['标识列'].iloc[0] in selected_data_1):
        #                     st.write('所选的炉次为:')
        #                     st.write(every_data)

        with st.expander("缺失值剔除"):
            if st.session_state.selected:
                for i in range(len(qsh_all)):
                    if (len(qsh_all[i]) > 1):
                        if (qsh_all[i][0] == "氧枪高度"):
                            qsh_yangqiang = list(set(LU_1 + qsh_all[i][1:]))
                            selected_data_1 = st.multiselect('缺失值剔除：{}({}个)'.format(qsh_all[i][0],
                                                                                          len(qsh_yangqiang)),
                                                             options=qsh_yangqiang)
                        else:
                            selected_data_1 = st.multiselect('缺失值剔除：{}({}个)'.format(qsh_all[i][0],
                                                                                          len(qsh_all[i]) - 1),
                                                             options=qsh_all[i][1:])
                        if selected_data_1:
                            for i in range(len(Data_ts)):
                                every_data = Data_ts[i]
                                if (every_data['标识列'].iloc[0] in selected_data_1):
                                    st.write('所选的炉次为:')
                                    st.write(every_data)

        with st.expander("异常值剔除"):
            if st.session_state.selected:
                for i in range(len(ych_all)):
                    if (len(ych_all[i]) > 1):
                        selected_data_1 = st.multiselect('异常值剔除：{}({}个)'.format(ych_all[i][0],
                                                                                      len(ych_all[i]) - 1),
                                                         options=ych_all[i][1:])
                        if selected_data_1:
                            for i in range(len(Data_ts)):
                                every_data = Data_ts[i]
                                if (every_data['标识列'].iloc[0] in selected_data_1):
                                    st.write('所选的炉次为:')
                                    st.write(every_data)

            if st.session_state.selected:
                selected_data_3 = st.multiselect('时刻异常：TSC时刻({}个)'.format(len(TSC_1)), options=TSC_1)
                if selected_data_3:
                    for i in range(len(Data_ts)):
                        every_data = Data_ts[i]
                        if (every_data['标识列'].iloc[0] in selected_data_3):
                            st.write('所选的炉次为:')
                            st.write(every_data)

            if st.session_state.selected:
                selected_data_4 = st.multiselect('时刻异常：TSO时刻({}个)'.format(len(TSO_1 + TSO_2 + TSO_3)),
                                                 options=TSO_1 + TSO_2 + TSO_3)
                if selected_data_4:
                    for i in range(len(Data_ts)):
                        every_data = Data_ts[i]
                        if (every_data['标识列'].iloc[0] in selected_data_4):
                            st.write('所选的炉次为:')
                            st.write(every_data)

        # with st.expander("异常值剔除"):
        #     if st.session_state.selected:
        #         selected_data_2 = st.multiselect('异常值处理：氧枪高度({}个)'.format(len(LU_2)), options=LU_2)
        #         if selected_data_2:
        #             for i in range(len(Data_ts)):
        #                 every_data = Data_ts[i]
        #                 if (every_data['标识列'].iloc[0] in selected_data_2):
        #                     st.write('所选的炉次为:')
        #                     st.write(every_data)
        #
        #     if st.session_state.selected:
        #         selected_data_3 = st.multiselect('时刻异常：TSC时刻({}个)'.format(len(TSC_1)), options=TSC_1)
        #         if selected_data_3:
        #             for i in range(len(Data_ts)):
        #                 every_data = Data_ts[i]
        #                 if (every_data['标识列'].iloc[0] in selected_data_3):
        #                     st.write('所选的炉次为:')
        #                     st.write(every_data)
        #
        #     if st.session_state.selected:
        #         selected_data_4 = st.multiselect('时刻异常：TSO时刻({}个)'.format(len(TSO_1 + TSO_2 + TSO_3)),
        #                                          options=TSO_1 + TSO_2 + TSO_3)
        #         if selected_data_4:
        #             for i in range(len(Data_ts)):
        #                 every_data = Data_ts[i]
        #                 if (every_data['标识列'].iloc[0] in selected_data_4):
        #                     st.write('所选的炉次为:')
        #                     st.write(every_data)

        if st.session_state.selected:
            jieguo_1, jieguo_2, jieguo_3 = st.columns([1, 1, 1])
            jieguo_1.write("高频数据筛选后的数据:")
            jieguo_2.write(len(get_bsl))

        def save_file_to_local_path_ts(file, local_path):
            file.to_csv(local_path, encoding='utf-8-sig')
            st.success(f"文件已保存到本地路径: {local_path}")

        def download_result_ts(df,local_name):
            # 将 DataFrame 转换为 CSV 格式的字节流
            csv = df.to_csv(index=False).encode('utf-8-sig')
            # 将字节流转换为 BytesIO 对象
            csv_io = BytesIO(csv)
            # 提供下载链接
            st.download_button(label="保存高频文件", data=csv_io, file_name=local_name, mime='text/csv')

        def download_result_all(df,local_name):
            # 将 DataFrame 转换为 CSV 格式的字节流
            csv = df.to_csv(index=False).encode('utf-8-sig')
            # 将字节流转换为 BytesIO 对象
            csv_io = BytesIO(csv)
            # 提供下载链接
            st.download_button(label="保存数据预处理后文件", data=csv_io, file_name=local_name, mime='text/csv')

        # 9、根据【吹炼开始时刻，25%吹氧进程，TSC时刻，TSO时刻】把每一炉次的高频数据分为3段，求均值及积分值。
        # 9.1、根据标识列，获取高频数据及其均值和积分值
        v_name_ts_add = v_name_ts + ['底吹气体流量1和2均值1', '底吹气体流量3和4均值2',
                                     '底吹气体流量5和6均值3', '底吹气体流量7和8均值4', '底吹气体流量9和10均值5',
                                     '底吹气体流量11和12均值6', '底吹气体流量13和14均值7',
                                     '底吹气体流量15和16均值8', '底吹气体流量均值', '底吹气体压力均值']
        data_ts_cal = pd.DataFrame(np.arange(0, len(v_name_ts_add))[np.newaxis, :], columns=v_name_ts_add)
        v_ts_lis = [v_name_ts_add.index('标识列'), v_name_ts_add.index('氧枪高度'), v_name_ts_add.index('氧气流量'),
                    v_name_ts_add.index('氧气压力'),
                    v_name_ts_add.index('气体压力1'), v_name_ts_add.index('气体压力2'),
                    v_name_ts_add.index('底吹气体流量均值'), v_name_ts_add.index('底吹气体压力均值')]
        v_name_ts_new = []
        for i in v_ts_lis[1:]:
            for j in ['吹炼前期均值', '吹炼中期均值', '吹炼后期均值', '吹炼全程均值']:
                # print(v_ts_name[i]+j)
                v_name_ts_new.append(v_name_ts_add[i] + j)
        Data_ts_mean = []
        Data_ts_mean_sta = []
        for i in range(len(Data_ts_real)):
            if Data_ts_real[i]['标识列'].iloc[0] in get_bsl:  # 判断炉次号是否符合
                t = Data_ts_len_real[i]  # 获得第i个炉次的时间长度
                # =====================================================================
                # 获得每个炉次内部'底吹气体流量均值', '底吹气体压力均值'，时间序列
                Data_align_2 = np.zeros((t, len(v_name_ts_add)))
                Data_align_2[:, :len(v_name_ts)] = Data_ts_real[i]
                k = 0
                for iii in range(v_name_ts.index('底吹气体流量1'), v_name_ts.index('气体压力1'), 2):
                    Data_align_2[:, len(v_name_ts) + k] = Data_ts_real[i].iloc[:, iii:iii + 2].sum(
                        axis=1)  # )[:,:,np.newaxis]
                    k += 1
                # '底吹气体流量均值'
                Data_align_2[:, len(v_name_ts) + 8] = Data_align_2[:, len(v_name_ts):(len(v_name_ts) + 8)].mean(axis=1)
                # '底吹气体压力均值'
                Data_align_2[:, len(v_name_ts) + 9] = Data_align_2[:, v_name_ts.index('气体压力1'):v_name_ts.index(
                    '气体压力1') + 2].mean(axis=1)
                Data_ts_mean.append(Data_align_2)
                # =====================================================================

                # =====================================================================
                # 获得v_name_ts_new    # ['氧枪高度吹炼前期均值', '氧枪高度吹炼中期均值','氧枪高度吹炼后期均值','氧枪高度吹炼全程均值',
                #  '氧气流量吹炼前期均值','氧气流量吹炼中期均值','氧气流量吹炼后期均值','氧气流量吹炼全程均值',
                #  '氧气压力吹炼前期均值','氧气压力吹炼中期均值','氧气压力吹炼后期均值','氧气压力吹炼全程均值',
                #  '底吹气体流量均值吹炼前期均值','底吹气体流量均值吹炼中期均值','底吹气体流量均值吹炼后期均值','底吹气体流量均值吹炼全程均值',
                #  '底吹气体压力均值吹炼前期均值','底吹气体压力均值吹炼中期均值','底吹气体压力均值吹炼后期均值',  '底吹气体压力均值吹炼全程均值',
                # '气体压力1吹炼前期均值', '气体压力1吹炼中期均值', '气体压力1吹炼后期均值', '气体压力1吹炼全程均值',
                # '气体压力2吹炼前期均值', '气体压力2吹炼中期均值', '气体压力2吹炼后期均值', '气体压力2吹炼全程均值']
                Data_ts_model = Data_align_2[:, v_ts_lis]  # Data_align_2对应列名v_name_ts_add
                sec = 4
                Data_ts_model1 = np.zeros((1, (len(v_ts_lis) - 1) * sec + 1))
                Data_ts_model1[:, 0] = Data_ts_real[i]['标识列'].iloc[0]  # 获得炉次号，即标识列
                sec1 = int(t * 0.25)
                sec2 = get_tsc[i].index.tolist()[0] - get_start[i]  # TSC分段点
                sec3 = get_tq[i].index.tolist()[0] - get_start[i]  # 氧枪提枪时刻分段点
                if sec2 < sec1:
                    print('第{}个炉次{},TSC时刻小于25%吹氧进程，求出nan值'.format(i, Data_ts_real[i]['标识列'].iloc[0]))
                    continue
                v_1 = np.arange(1, Data_ts_model1.shape[1] + 3, sec)  # [ 1,  4,  7, 10, 13]
                for i_tscal in range(len(v_ts_lis) - 1):
                    Data_ts_model1[:, v_1[i_tscal]] = Data_ts_model[:sec1, i_tscal + 1].mean(axis=0)  # 吹炼前期
                    Data_ts_model1[:, v_1[i_tscal] + 1] = Data_ts_model[sec1:sec2, i_tscal + 1].mean(axis=0)  # 吹炼中期
                    Data_ts_model1[:, v_1[i_tscal] + 2] = Data_ts_model[sec2:sec3, i_tscal + 1].mean(axis=0)  # 吹炼后期
                    Data_ts_model1[:, v_1[i_tscal] + 3] = Data_ts_model[:sec3, i_tscal + 1].mean(axis=0)  # 吹炼全程
                Data_ts_model2 = pd.DataFrame(Data_ts_model1, columns=['标识列'] + v_name_ts_new)

                # =====================================================================

                # =====================================================================
                # 获得曲线下方面积（积分求得）(氧气流量，氮气、氩气)# 3个阶段的耗氧量、耗氮量和耗氩量
                Data_align_2 = pd.DataFrame(Data_align_2, columns=v_name_ts_add)
                Data_align_2['吹炼前期耗氧量'] = np.trapz(y=Data_align_2['氧气流量'].iloc[:sec1],
                                                          x=range(0, sec1))  # 求曲线下方的面积，利用定积分
                Data_align_2['吹炼中期耗氧量'] = np.trapz(y=Data_align_2['氧气流量'].iloc[sec1:sec2],
                                                          x=range(sec1, sec2))  # 求曲线下方的面积，利用定积分
                Data_align_2['吹炼后期耗氧量'] = np.trapz(y=Data_align_2['氧气流量'].iloc[sec2:sec3],
                                                          x=range(sec2, sec3))  # 求曲线下方的面积，利用定积分
                Data_align_2['吹炼全程耗氧量'] = np.trapz(y=Data_align_2['氧气流量'].iloc[:sec3],
                                                          x=range(0, sec3))  # 求曲线下方的面积，利用定积分

                Data_align_2['吹炼前期耗氩量'] = np.sum(
                    [np.trapz(y=Data_align_2['底吹气体流量{}'.format(ll_num)].iloc[:sec1], x=range(0, sec1)) for ll_num
                     in range(1, 16, 2)])
                Data_align_2['吹炼中期耗氩量'] = np.sum(
                    [np.trapz(y=Data_align_2['底吹气体流量{}'.format(ll_num)].iloc[sec1:sec2], x=range(sec1, sec2)) for
                     ll_num in range(1, 16, 2)])
                Data_align_2['吹炼后期耗氩量'] = np.sum(
                    [np.trapz(y=Data_align_2['底吹气体流量{}'.format(ll_num)].iloc[sec2:], x=range(sec2, t)) for ll_num
                     in range(1, 16, 2)])
                Data_align_2['吹炼全程耗氩量'] = np.sum(
                    [np.trapz(y=Data_align_2['底吹气体流量{}'.format(ll_num)].iloc[:], x=range(0, t)) for ll_num in
                     range(1, 16, 2)])

                Data_align_2['吹炼前期耗氮量'] = np.sum(
                    [np.trapz(y=Data_align_2['底吹气体流量{}'.format(ll_num)].iloc[:sec1], x=range(0, sec1)) for ll_num
                     in range(2, 17, 2)])
                Data_align_2['吹炼中期耗氮量'] = np.sum(
                    [np.trapz(y=Data_align_2['底吹气体流量{}'.format(ll_num)].iloc[sec1:sec2], x=range(sec1, sec2)) for
                     ll_num in range(2, 17, 2)])
                Data_align_2['吹炼后期耗氮量'] = np.sum(
                    [np.trapz(y=Data_align_2['底吹气体流量{}'.format(ll_num)].iloc[sec2:], x=range(sec2, t)) for ll_num
                     in range(2, 17, 2)])
                Data_align_2['吹炼全程耗氮量'] = np.sum(
                    [np.trapz(y=Data_align_2['底吹气体流量{}'.format(ll_num)].iloc[:], x=range(0, t)) for ll_num in
                     range(2, 17, 2)])

                df_sum = Data_align_2.iloc[0:1, -12:]
                Data_ts_cal_all = Data_ts_model2.join(df_sum)
                Data_ts_mean_sta.append(Data_ts_cal_all)

        Data_ts_mean_sta_final = pd.DataFrame(np.array(Data_ts_mean_sta).squeeze(),
                                              columns=Data_ts_cal_all.columns.tolist())
        bsl_new = pd.DataFrame(np.array(get_bsl), columns=['标识列'])
        tsc_time_new = pd.DataFrame(np.array(get_tsc_time), columns=['TSC时刻'])
        bsl_time = pd.concat([bsl_new, tsc_time_new], axis=1)
        data_sta_final = pd.merge(bsl_time, Data_sta2, on='标识列')
        data_final = pd.merge(data_sta_final, Data_ts_mean_sta_final, on='标识列')

        if st.session_state.selected:
            col13, col14, col15 = st.columns([1, 1, 1])
            write_data_button_ts = col13.button("显示高频数据处理后结果")
            if write_data_button_ts:
                st.write(Data_ts_mean_sta_final)
                st.session_state.selected_1 = True

        # if st.session_state.selected_1:
        #     col16, col17, col18 = st.columns([1, 1, 1])
        #     save_data_button_ts = col16.button("保存高频文件")
        #     local_path1 = st.text_input("高频数据的保存路径（包括文件名.csv）:",
        #                                 os.path.join(file_path,
        #                                              '炉次数为{}_{}-{}.csv'.format(
        #                                                  Data_ts_mean_sta_final.shape[0],
        #                                                  int(Data_ts_mean_sta_final.标识列[0]),
        #                                                  int(Data_ts_mean_sta_final['标识列'].iloc[-1])
        #                                              )))
        #     if save_data_button_ts:
        #         save_file_to_local_path_ts(Data_ts_mean_sta_final, local_path1)

        if st.session_state.selected_1:
            local_name1 = st.text_input("高频数据文件保存（文件名.csv）:",
                                        '高频处理后的炉次{}.csv'.format(Data_ts_mean_sta_final.shape[0]))
            download_result_ts(Data_ts_mean_sta_final,local_name1)

        if st.session_state.selected:
            col19, col20, col21 = st.columns([1, 1, 1])
            write_data_button_all = col19.button("显示数据预处理后结果")
            if write_data_button_all:
                st.write(data_final)
                st.session_state.selected_2 = True

        if st.session_state.selected_2:
            local_name2 = st.text_input("数据预处理后文件保存（文件名.csv）:",
                                        '数据预处理后的炉次{}.csv'.format(data_final.shape[0]))
            download_result_all(data_final,local_name2)

# 在侧边栏中创建radio组件
base_selector = st.sidebar.radio('界面', ('1-数据载入',
                                          '2-低频数据预处理', '3-高频数据预处理','4-选择变量',))

# 根据选择的基面展示不同的表格
if base_selector == '1-数据载入':
    show_base1()
elif base_selector == '4-选择变量':
    show_base2()
elif base_selector == '2-低频数据预处理':
    show_base3()
elif base_selector == '3-高频数据预处理':
    show_base4()
