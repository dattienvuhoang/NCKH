import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import chardet
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)
df_cluster = None
data_file = None
df = None
radio_selection = None
n = 1


def read_csv_with_file_uploader():
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        # To detect file encoding
        encoding = chardet.detect(bytes_data)['encoding']
        # To convert to a string based IO:
        stringio = io.StringIO(bytes_data.decode(encoding))
        # To read file as string (assuming it's a CSV file):
        df = pd.read_csv(stringio)
        return df


def input_file(data_file, n, radio_selection, df_cluster):
    global df
    global selected_columns_list
    data_file = read_csv_with_file_uploader()
    if data_file is not None:
        df = data_file.dropna()
        df = df.head(2000)
        st.dataframe(data_file)
        st.write('Loại bỏ các giá trị Null:')
        st.dataframe(df)
        st.write('Tổng quan dữ liệu:')
        st.dataframe(data_file.describe())
        selected_columns = st.multiselect('Lựa chọn dữ liệu phân cụm:', df.columns.to_list())
        if len(selected_columns) >= 4:
            st.error('Chỉ lựa chọn tối đa 3 cột dữ liệu để phân cụm.')
            return
        selected_columns_list = list(selected_columns)
        if selected_columns:
            df_cluster = pd.DataFrame(df[selected_columns_list])
            st.dataframe(df_cluster)
            Elbow(df_cluster)
            if radio_selection == 'K-MEANS':
                n = int(st.number_input('Nhập số cụm', min_value=2, key=int))
                df_cluster = runKmean(df_cluster, n)
            else:
                df_cluster = runDbScan(df_cluster)
    return df_cluster


def export_clustered_data():
    if df is not None:
        data = df.sort_values('Cluster')
        output_filename = 'clustered_data.csv'
        data_csv = data.to_csv(index=False)
        if data_csv:
            st.download_button(label='TẢI VỀ KẾT QUẢ PHÂN CỤM',
                               data=data_csv,
                               file_name=output_filename)
            st.dataframe(data)
        else:
            st.write('No data to export.')


def runKmean(df_cluster, n):
    st.title('Biểu đồ phân cụm')
    global selected_columns_list
    if df_cluster is not None:
        kmeans = KMeans(
            n_clusters=n, init='k-means++', max_iter=300, n_init=10
        )
        clusters = kmeans.fit_predict(df_cluster)
        df_cluster['Cluster'] = kmeans.labels_
        centroids = kmeans.cluster_centers_
        cluster_counts = df_cluster['Cluster'].value_counts()
        if len(selected_columns_list) > 2:
            # Create a 3D scatter plot of the clusters
            fig = go.Figure()
            # Define a color palette for the clusters
            colors = px.colors.qualitative.Plotly
            # Add scatter plot for clusters
            for i in range(n):
                cluster_df = df_cluster[df_cluster['Cluster'] == i]
                fig.add_trace(go.Scatter3d(
                    x=cluster_df[selected_columns_list[0]],
                    y=cluster_df[selected_columns_list[1]],
                    z=cluster_df[selected_columns_list[2]],
                    mode='markers',
                    marker=dict(size=3, color=colors[i % len(colors)]),
                    name=f'Cluster {i}'
                ))
                # Add lines from centroid to each point in the cluster
                for _, row in cluster_df.iterrows():
                    fig.add_trace(go.Scatter3d(
                        x=[centroids[i][0], row[selected_columns_list[0]]],
                        y=[centroids[i][1], row[selected_columns_list[1]]],
                        z=[centroids[i][2], row[selected_columns_list[2]]],
                        mode='lines',
                        line=dict(color=colors[i % len(colors)], width=2),
                        showlegend=False
                    ))

            # Add scatter plot for centroids
            fig.add_trace(go.Scatter3d(
                x=centroids[:, 0],
                y=centroids[:, 1],
                z=centroids[:, 2],
                mode='markers',
                marker=dict(size=6, color='black'),
                name='Centroids'
            ))

            # Update layout for a better view
            fig.update_layout(
                scene=dict(
                    xaxis_title=selected_columns_list[0],
                    yaxis_title=selected_columns_list[1],
                    zaxis_title=selected_columns_list[2]
                ),
                legend=dict(
                    title='',
                    itemsizing='constant'
                )
            )

            # Display the figure in Streamlit
            st.plotly_chart(fig)
            st.write('Số lượng điểm dữ liệu trong mỗi cụm:', cluster_counts)
        else:

            # Tạo biểu đồ phân tán với các điểm dữ liệu được tô màu theo cụm
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(
                df_cluster.iloc[:, 0],
                df_cluster.iloc[:, 1],
                c=clusters,
                cmap='viridis',
                marker='o'
            )
            # Đánh dấu tâm cụm
            centers = kmeans.cluster_centers_
            center_scatter = plt.scatter(
                centers[:, 0],
                centers[:, 1],
                c='red',
                s=200,
                alpha=0.75,
                marker='x'
            )
            plt.xlabel(selected_columns_list[0])
            plt.ylabel(selected_columns_list[1])
            # Thêm chú thích cho biểu đồ phân cụm
            plt.legend(*scatter.legend_elements(), title='Clusters')
            # Hiển thị biểu đồ trên Streamlit
            st.pyplot()
            st.write('Số lượng điểm dữ liệu trong mỗi cụm:', cluster_counts)
    return df_cluster


# Dựa theo dữ liệu đầu vào, phân tích và đưa ra giá trị eps và min_samples tối ưu
def find_optimal_eps_min_samples(df_cluster):
    from sklearn.neighbors import NearestNeighbors
    from kneed import KneeLocator

    # Tìm số lượng hàng xóm gần nhất cho mỗi điểm dữ liệu
    nearest_neighbors = NearestNeighbors(n_neighbors=2)
    neighbors = nearest_neighbors.fit(df_cluster)
    distances, indices = neighbors.kneighbors(df_cluster)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    # Tìm giá trị eps tối ưu
    kneedle = KneeLocator(
        range(1, len(distances) + 1), distances, curve='convex', direction='increasing'
    )
    eps = distances[kneedle.elbow]
    # Tìm giá trị min_samples tối ưu
    min_samples = 2 * df_cluster.shape[1]
    st.write('Giá trị eps tối ưu:', eps)
    st.write('Giá trị min_samples tối ưu:', min_samples)
    return eps, min_samples


def runDbScan(df_cluster):
    global selected_columns_list
    radio_button = st.radio('Lựa chọn giá trị eps và min_samples', ['Tối ưu', 'Tự nhập'])
    if radio_button == 'Tối ưu':
        eps, min_samples = find_optimal_eps_min_samples(df_cluster)
    else:
        eps = st.slider('Chọn giá trị eps', min_value=0.1, max_value=100.0, value=0.1, step=0.1)
        min_samples = st.slider('Chọn giá trị min_samples', min_value=1, max_value=200, value=5, step=1)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(df_cluster)
    df_cluster['Cluster'] = clusters
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Tạo biểu đồ phân tán với các điểm dữ liệu được tô màu theo cụm
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(
        df_cluster.iloc[:, 0],
        df_cluster.iloc[:, 1],
        c=clusters,
        cmap='viridis',
        marker='o'
    )
    # Add labels for cluster names and noise
    for i, txt in enumerate(clusters):
        if txt == -1:
            plt.annotate('', (df_cluster.iloc[i, 0], df_cluster.iloc[i, 1]))
        else:
            plt.annotate(f'{txt}', (df_cluster.iloc[i, 0], df_cluster.iloc[i, 1]))

    plt.title('DBSCAN Clustering')
    plt.xlabel(df_cluster.columns[0])
    plt.ylabel(df_cluster.columns[1])
    st.pyplot(fig)
    # Count the number of data points in each cluster, excluding noise points
    cluster_counts = df_cluster['Cluster'].value_counts()
    cluster_counts = cluster_counts[cluster_counts.index != -1]  # Exclude noise points
    # Count the number of noise points
    n_noise = list(clusters).count(-1)
    st.write('Số lượng cụm:', len(cluster_counts))
    st.write('Số lượng điểm nhiễu:', n_noise)
    # Display the number of data points in each cluster
    st.write('Số lượng điểm dữ liệu trong mỗi cụm:')
    st.dataframe(cluster_counts)
    return df_cluster


def Elbow(df_cluster):
    st.title('Chọn số cụm tối ưu bằng phương pháp Elbow')
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(df_cluster)
        wcss.append(kmeans.inertia_)
    # Plot the elbow graph
    st.set_option('deprecation.showPyplotGlobalUse', False)  # To avoid deprecation warning
    fig, ax = plt.subplots(figsize=(10, 5))  # Create a new figure with a defined axis
    ax.plot(range(1, 11), wcss, marker='o', linestyle='--')
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)


def run():
    global df_cluster
    st.set_page_config(
        page_title='Demo Sản Phẩm',
        page_icon='💻',
    )

    with st.sidebar:
        st.title('Menu')
        radio_selection = st.radio('Lựa chọn thuật toán', ['K-MEANS', 'DBSCAN'])
    st.title('Nghiên cứu một số vấn đề về dữ liệu lớn và học máy, ứng dụng trong phân loại khách hàng')
    if radio_selection == 'K-MEANS':
        st.markdown("<h1 style='text-align: center;'>KMEAN CLUSTERING</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: center;'>DBSCAN CLUSTERING</h1>", unsafe_allow_html=True)
    df_cluster = input_file(data_file, n, radio_selection, df_cluster)
    if df_cluster is not None and 'Cluster' in df_cluster.columns:
        df['Cluster'] = df_cluster['Cluster']
        export_clustered_data()


print('Running main func...')
# running main func
if __name__ == '__main__':
    run()
