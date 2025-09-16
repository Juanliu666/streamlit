import requests
import streamlit as st

# 设置页面布局
st.set_page_config(layout="wide")

# 初始化数据存储
if 'result' not in st.session_state:
    st.session_state.result = None

# 页面标题
st.title("污泥产物预测系统")
st.text("本系统基于注意力机制-长短时记忆网络算法（Attention-lstm）开发")

# 创建两列布局
col1, col2 = st.columns(2)

# 左侧输入区域
with col1:
    st.header("输入参数")
    input1 = st.text_input("污泥添加比例（%）", key="input1")
    input2 = st.text_input("C含量（%）", key="input2")
    input3 = st.text_input("热解温度（℃）", key="input3")

    # 创建按钮并设置样式
    if st.button("预测", key="predict_button"):
        with st.spinner("预测中..."):
            # 按钮点击后的处理逻辑
            if input1 and input2 and input3:
                data = {"污泥添加比例（%）": input1, "C含量（%）": input2, "热解温度（℃）": input3}
                try:
                    req = requests.post('http://localhost:5000/predict', data=data)
                    st.session_state.result = req.json()
                except:
                    st.error("无法连接到预测服务，请检查后端是否正常运行")
            else:
                st.warning("请填写所有输入框")

# 设置预测按钮的样式
st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #0066cc;
        color: white;
    }
    div.stButton > button:hover {
        background-color: #0052a3;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 右侧输出区域
with col2:
    st.header("预测结果")

    # 结果展示区域 - 添加蓝色背景
    if st.session_state.result:
        # 使用自定义CSS设置结果区域的背景色
        st.markdown(
            """
            <style>
            .result-container {
                background-color: #e6f2ff;
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # 将结果放在带背景色的容器中
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        st.success("数据预测完成！")

        # 显示预测结果
        for key, value in st.session_state.result["predictions"].items():
            st.markdown(f"<div style='margin: 10px 0;'><strong>{key}:</strong> {value}</div>",
                        unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # 原始数据查看
        with st.expander("查看原始JSON数据"):
            st.json(st.session_state.result)
    else:
        st.info("请输入参数并点击预测按钮获取结果")

