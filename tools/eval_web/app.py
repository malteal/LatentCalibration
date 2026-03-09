
import dash
import dash_core_components as dcc
import plotly.graph_objects as go
import dash_html_components as html
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash.exceptions import PreventUpdate
import dash_daq as daq
# import dash_bootstrap_components as dbc

from glob import glob
import numpy as np
import pandas as pd
import os
import sys
import json
import logging
import dash_table as dt
from tqdm import tqdm
from tools import misc


# init logging 
logging.basicConfig(
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger(__name__)

app = dash.Dash(
    __name__, 
    # external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width"}],
    assets_folder = './assets/',
    update_title=None,
    )

app.title = "W&B"
server = app.server
app.config.suppress_callback_exceptions = True


def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        # id="description-card",
        children=[
            html.Br(),
            html.H5("Stupid WandB", style={'color':'#2c8cff'}),
            html.H3("Welcome to the Stupid WandB"),
            html.Div(
                    ["""
                     Weight and bias alike
                     """,
                    html.Br(),
                    html.Br(),
                    "Problem with the website?",
                    html.Br(),
                    "Contact: malte.algren@unige.ch"
                    ],
                # id="intro",
            ),
        ]
    )

app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("2560px-Uni_GE_logo.svg.png"), style={'height': '150%'})],
        ),
        ######## Left column ########
        html.Div(
            id="left-column",
            className="four columns",
            children=
                    [description_card()]+
                    [html.Div([dcc.Input(
                            id="overall_path",
                            type="text",
                            value = "/home/users/a/algren/scratch/trained_networks/decorrelation/01_22_2023_08_57_22_391332/",
                            placeholder="Input absolute path",
                            className="nine columns",
                        ),
                    html.Button('Submit path', id='submit_overall_path', n_clicks=0,
                                className="three columns"),
                    ], className="row"),
                    html.Br(),
                    html.Label("Select runs"),
                    html.Div([
                    dcc.Dropdown(
                        id="dropdown_path_selected",
                        multi=True,
                        className="ten columns"
                    ),
                    dcc.Input(
                            id="input_nr_figures",
                            type="number",
                            value = 1,
                            placeholder="Number of figures",
                            className="two columns",
                    )
                    ], className="row")
                    ]
                    + [html.Div(
                        id="select_element",
                        )],
        ),
        ######## Right column ########
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
            ],
        ),
        html.Div([
            html.Br(),
            html.Br(),
            dt.DataTable(
            id='summary_datatable',
            data= [],
            columns=[],
            row_selectable="multi",
            sort_action="native",
            sort_mode="multi",
            page_action="native",
            page_current= 0,
            page_size= 10,
            ),
            html.Br(),
            html.Br(),
            ], className="twelve columns"),
    # dummy callback
    html.Div([
        dcc.Interval(id="interval-log-update", n_intervals=1000),
        # store data
        dcc.Store(id='memory-dropdown', storage_type='session', data={}),
        dcc.Store(id="memory-plots", storage_type="session", data={}),
    ])
    ],
)

def get_selected_rows_path(table_paths, summary_table, global_path):
    paths_from_tables=[]
    for row in table_paths:
        paths_from_tables.append(global_path+"/"+summary_table[row]["name"])
    return paths_from_tables

def handle_memory(memory:dict, name:list, value:list=None, delete=False):
    """Fill out memory
        No nested dict!!!!
    """
    if not isinstance(name, list):
        raise TypeError("Name has to be a list")

    if value is None:
        value = []
        for i in name:
            _value = memory.setdefault(i, None)
            value.append(_value)
    elif delete:
        for i in name:
            memory.pop(i, None)
    else:
        for val, i in zip(value, name):
            memory[i] = val
    return memory, value

@app.callback(
    Output("interval-log-update", "interval"),
    [Input("dropdown-interval-control", "value")],
)
def update_interval_log_update(interval_rate):
    if interval_rate == "fast":
        return 500

    elif interval_rate == "regular":
        return 1000

    elif interval_rate == "slow":
        return 5 * 1000

    # Refreshes every 24 hours
    elif interval_rate == "no":
        return 24 * 60 * 60 * 1000

@app.callback(
    Output("dropdown_path_selected", "options"),
    Output("dropdown_path_selected", "value"),
    Output("memory-dropdown", 'data'),
    Input("submit_overall_path", "n_clicks"),
    State("overall_path", "value"),
    State("memory-dropdown", 'data'),
    State("memory-plots", 'data'),
    verbose=True
    )
def update_dropdown_options(n_clicks, path, memory_dropdown, memory_plots):
    # check memory
    path = path.replace("*", "")
    _, path_selected = handle_memory(memory=memory_plots,
                                     name=["dropdown_path_selected"])
    memory_dropdown, value = handle_memory(memory=memory_dropdown,
                                           name=["update_dropdown_options",
                                                 "n_clicks"])
    # logger.info(value[0])
    if (value[0] is not None) & (n_clicks==value[1]):
        return value[0], path_selected[0], memory_dropdown

    # check default function
    if n_clicks > 0:
        collections_runs = sorted(glob(path+'/*'), key=os.path.getctime)[::-1]
        options = [{"label": i.split("/")[-1], "value": i} for i in collections_runs]
        if path_selected[0] is not None:
            path_selected = np.intersect1d(path_selected[0], collections_runs)
        return options,path_selected, handle_memory(memory=memory_dropdown,
                                         name=["update_dropdown_options","n_clicks"],
                                         value=[options, n_clicks])[0]
    else:
        raise PreventUpdate

@app.callback(
    Output("input_nr_figures", "value"),
    Input("dropdown_path_selected", "value"),
    State("memory-plots", 'data'),
    verbose=True
    )
def _correct_n_plots(_, memory_plots):
    "Made to have the input_nr_figures number correct"
    return  memory_plots.get("n_plots", 1)

@app.callback(
    Output("select_element", "children"),
    Output("right-column", "children"),
    Output("memory-plots", 'data'),
    Input("dropdown_path_selected", "value"),
    Input("summary_datatable", "selected_rows"),
    Input("input_nr_figures", "value"), 
    State({'type': 'dynamic_select_variables', 'index': dash.ALL}, 'options'),
    State({'type': 'dynamic_select_variables', 'index': dash.ALL}, 'value'),
    State({'type': 'dynamic_plot', 'index': dash.ALL}, 'figure'),
    State("submit_overall_path", "n_clicks"),
    State("summary_datatable", 'data'),
    State("overall_path", 'value'),
    State("memory-dropdown", 'data'),
    State("memory-plots", 'data'),
    verbose=True
    )
def number_of_plots(i_dropdown_path, i_table_paths, input_nr_figures, s_old_dropdowns_options, 
                    s_old_dropdowns_values, old_figures, submit_overall_path,
                    summary_table, global_path,
                    memory_dropdown, memory_plots):
    """
    :return: A Div containing controls for graphs.
    """
    global_path = global_path.replace("*", "")
    memory_plots, value = handle_memory(memory=memory_plots, name=["number_of_plots",
                                                                   "dropdown_path_selected"])
    # print(memory_dropdown)
    if ((value[0] is not None)
        & (input_nr_figures==memory_plots.get("n_plots", -999))
        & (memory_dropdown["n_clicks"]!=submit_overall_path)):
        return value[0][0], value[0][1], memory_plots
    paths = []

    #selected from table
    # try:
    if i_table_paths is not None:
        paths_from_tables = get_selected_rows_path(i_table_paths, summary_table, global_path)
        paths.extend(paths_from_tables)
    # except TypeError as e:
    #     print("Not using gridsearch, so table do not need plotting!")
    
    #selected from dropdown
    if i_dropdown_path is not None:
        paths.extend(i_dropdown_path)

    if (paths == []) or (paths[0] is None):
        # no path present
        raise PreventUpdate

    children = []
    right_children = []
    if isinstance(paths, list) and len(paths)>0:
        logger.debug(f"Running paths {paths}")
        path_to_log = glob(paths[0]+"/lo*.json")
        print(paths)
        with open(path_to_log[0], "r+") as fp:
            data = json.load(fp)
        all_dict_keys = misc.get_dict_keys(data)
        default_options = [{"label": i.replace("...", "."), "value": i}
                           for i in all_dict_keys]
        for nr in range(input_nr_figures):
            # output graph
            # select old or None figure
            if nr < len(old_figures):
                figure = old_figures[nr]
            else:
                figure = {}
            right_children.extend([dcc.Graph(id={
                                                    'type': 'dynamic_plot',
                                                    'index': nr
                                                },
                                            figure=figure
                                            ),
                                  html.Br()])
            # select dropdown values and options
            print("number ", nr)
            print("s_old_dropdowns_options ", len(s_old_dropdowns_options))
            print("s_old_dropdowns_values ", len(s_old_dropdowns_values))
            value = None
            if nr < len(s_old_dropdowns_values):
                value = s_old_dropdowns_values[nr]
                
            options = default_options
            if nr < len(s_old_dropdowns_options):
                options = s_old_dropdowns_options[nr]

            # output the dropdowns
            children.extend(
                    [html.H6(f"{nr}: Select variables")]+
                    [dcc.Dropdown(
                        id={
                                'type': 'dynamic_select_variables',
                                'index': nr
                            },
                        options=options,
                        value = value,
                        multi=True,
                    )]
                    +[html.Div(
                    children=[
                            daq.BooleanSwitch(
                                label='Y log scale',
                                labelPosition='top',
                                on=False,
                                id={
                                'type': 'y_scale_toggle',
                                'index': nr
                                    },
                                className="three columns",
                            ),
                            daq.BooleanSwitch(
                                label='X log scale',
                                labelPosition='top',
                                on=False,
                                id={
                                'type': 'x_scale_toggle',
                                'index': nr
                                    },
                                className="three columns",
                            ),
                            daq.BooleanSwitch(
                                label='Vertical lines',
                                labelPosition='top',
                                on=False,
                                id={
                                'type': 'v_lines_toggle',
                                'index': nr
                                },
                                className="three columns",
                            ),
                            daq.BooleanSwitch(
                                label='Max / Min',
                                labelPosition='top',
                                on=False,
                                id={
                                'type': 'max_min_toggle',
                                'index': nr
                                },
                                className="three columns",
                            ),
                    ], className="row"
                    ),
                    ]+[html.Br() for _ in range(5)]
                    )

    memory_plots = handle_memory(memory=memory_plots,
                                 name=["n_plots",
                                       "dropdown_path_selected",
                                       "number_of_plots"],
                                 value=[input_nr_figures,
                                        i_dropdown_path,
                                        [children, right_children]])[0]
    
    return children, right_children, memory_plots

@app.callback(
    Output({"type":"dynamic_plot", "index": dash.MATCH}, "figure"),
    Input({"type":"dynamic_select_variables", "index": dash.MATCH}, "value"),
    Input({"type":"x_scale_toggle", "index": dash.MATCH}, "on"),
    Input({"type":"y_scale_toggle", "index": dash.MATCH}, "on"),
    Input({"type":"max_min_toggle", "index": dash.MATCH}, "on"),
    Input({"type":"v_lines_toggle", "index": dash.MATCH}, "on"),
    State("dropdown_path_selected", "value"),
    State("summary_datatable", "selected_rows"),
    State("summary_datatable", 'data'),
    State("overall_path", 'value'),
    verbose=True
    )
def plot_variables(variables,  x_scale_toggle, y_scale_toggle, min_max, v_lines_toggle,
                   path_to_log, table_paths, summary_table, global_path):
    global_path = global_path.replace("*", "")
    paths=[]
    if table_paths is not None:
        paths_from_tables = get_selected_rows_path(table_paths, summary_table, global_path)
        #selected from table
        if paths_from_tables is not None:
            paths.extend(paths_from_tables)
    #selected from dropdown
    if path_to_log is not None:
        paths.extend(path_to_log)
    if (paths == []) or (variables == []) or (variables == None):
        raise PreventUpdate
    
    fig = go.Figure()
    # print(paths)
    for _path in paths:
        if _path is None: # dont know where the None is from
            continue
        path = glob(_path+"/lo*.json")
        if len(path)==0:
            print(f"Path: {_path} do not have that log")
            continue
        with open(path[0], "r+") as fp:
            data = json.load(fp)

            
        for var in variables:
            try:
                values = misc.get_data_from_dict(data, var.split("..."))
            except KeyError:
                print("Following key not working in def plot_variables")
                print(var)
                print(var.split("..."))
                continue
            # indicate max or min
            if min_max:
                min_max_point = np.argmin(values)
                min_max_str = "Min"
            else:
                min_max_point = np.argmax(values)
                min_max_str = "Max"
            try:
                value_min_max = f"{min_max_str}: {str(round(values[min_max_point],4))}"
            except KeyError:
                print("problems with value_min_max - setting it to Hello")
                value_min_max = "Hello"
            # plot variables
            if "steps" in data:
                x = data["steps"]
                xaxis_title = "Steps"
            else:
                x = np.arange(0, len(values))
                xaxis_title="Epochs"
            
            fig.add_trace(go.Scatter(x=x, y=values, name=f"{path[0].split('/')[-2]} - {value_min_max}"))
            if v_lines_toggle:
                fig.add_vline(min_max_point, line_width=3, line_dash="dash", line_color="green")
    if x_scale_toggle:
        fig.update_xaxes(type="log")
    if y_scale_toggle:
        fig.update_yaxes(type="log")

    print(var)
    
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=var.split("...")[-1],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=20, b=20),
                    )
    return fig

def get_json_files(original_path):
    "should use dict instead"
    data = []
    for i in ["/lo*.*", "/train_*.*", "/model_*.*", "/flow_*.*"]:
        path = glob(original_path+i)
        print(path)
        if len(path)==0:
            data.append([])
        elif "json" in path[0]:
            data.append(misc.load_json(path[0]))
        elif "yaml" in path[0]:
            data.append(misc.load_yaml(path[0]))
            
    return data[0], data[1], data[2]

def unpack_log(log, train_log, model_log, original_path, min_max_toggle):
    log_columns = [
        "log_likelihood_eval",
        "AUC",
        "source_average_wasserstein",
        "loss",
        "total...source...average_wasserstein",
        ]
    # log_columns = ["log_likelihood_eval"]
    log_model = ["convex_layersizes", "nonconvex_layersizes", "n_layers",
                 "convex_activation", "nonconvex_activation", "cvx_norm",
                 "noncvx_norm", "correction_trainable"]
    log_train = ["loss_wasser_ratio", "loss_li_ratio", "lr_f", "lr_g",
                 "f_per_g", "g_per_f", "batch_size"]

    content=[original_path.split("/")[-1]]
    for columns, data in zip([log_columns, log_model, log_train], [log, model_log, train_log]):
        for i in columns:
            try:
                # if not isinstance(log[i], list):
                #     continue
                if (i not in data) and ("..." not in i):
                    print(i)
                    content.append(None)
                elif "convex_layersizes" == i:
                    # print(data["convex_layersizes"])
                    content.append(data[i])
                elif "n_layers" == i:
                    content.append(data[i] if isinstance(data[i], int) else data[i][2])
                elif "optimizer_name" == i:
                    content.append(data[i]["f"])
                elif i in log_columns:
                    value = values = misc.get_data_from_dict(data, i.split("..."))
                    content.append(np.round(np.max(value) if not min_max_toggle
                                else np.min(value),3))
                elif isinstance(data[i], int) and not isinstance(data[i], bool):
                    content.append(np.round(np.max(data[i]) if not min_max_toggle
                                else np.min(data[i]),3))
                else:
                    if isinstance(data[i], str) or isinstance(data[i], bool):
                        content.append(data[i])
                    else:
                        content.append(np.round(np.max(data[i]) if not min_max_toggle
                                    else np.min(data[i]),5))
            except:
                content.append(None)
    return content, ["name"]+log_columns+log_model+log_train


@app.callback(
    [Output("summary_datatable", "data"),
    Output("summary_datatable", "columns")],
    Input("submit_overall_path", "n_clicks"),
    # Input("dropdown_path_selected", "value"),
    # Input({"type":"max_min_toggle", "index": dash.ALL}, "on"),
    State("overall_path", "value"),
    # State("summary_table", "children"),
    # State("output_datatable", "columns"),
    verbose=True
    )
def generate_table(
                   n_clicks,
                #    min_max_toggle,
                   overall_path,
                #    table_data,
                   ):
    min_max_toggle= True
    if n_clicks > 0:
        if "*" in overall_path:
            path_to_log = glob(overall_path)
        else:
            path_to_log = glob(overall_path+"/*")
        overall_content = []
        if len(path_to_log)==0:
            logging.INFO(f"Could not find folders in {overall_path}")
            raise PreventUpdate
        
        for original_path in tqdm(path_to_log):
            try:
                log, train_log, model_log = get_json_files(original_path)
            except (IndexError, json.decoder.JSONDecodeError) as e:
                print(e)
                print(f"Error in loading the json file from path {original_path}")
                continue
            content, header = unpack_log(log, train_log, model_log, original_path, min_max_toggle)
            overall_content.append(content)

        print(overall_content)
        df = pd.DataFrame(overall_content, columns=header)
        columns = [{"name": i, "id": i} for i in df.columns]

        df = df.to_dict('records')
        return df, columns

# /home/users/a/algren/scratch/trained_networks/GVAE/OT_gvae_gridsearch_09_18_2022_23_24_34_738276/
if __name__ == "__main__":
    print("Running local host")

    style = {"debug":False, "host":"0.0.0.0", "port":8081, "use_reloader":False,
            "dev_tools_hot_reload":False}

    app.run_server(**style)





