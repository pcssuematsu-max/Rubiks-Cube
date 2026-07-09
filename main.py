from ui.frame import Frame, build_default_bootstrap_datas
from ui.frame_config import FrameConfig

def _default_initial_scramble_groups(size,puzzle_type):
    """起動時に登録する既定の scramble 候補群を返す。"""
    if puzzle_type == "square1":
        return (
            [
                ((0, 0, "/"),),
                ((1, 0, None),),
                ((0, 1, None),),
                ((1, 1, "/"),),
                ((3, -2, "/"),),
                ((-3, 3, "/"),),
                ((-2, 0, "/"),),
                ((0, 3, "/"),),
                ((1, 1, "/"),(6,6,None)),
                ((3, -2, "/"),(6,6,None)),
                ((-3, 3, "/"),(6,6,None)),
                ((-2, 0, "/"),(6,6,None)),
            ],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    if puzzle_type == "skewb":
        return (
            [
                ("URF",),
                ("ULB",),
                ("UBR","ULB'","UBR'","ULB","UFL","URF'","UFL'","URF"),
                ("UFL'", 'ULB', "UFL'", "URF'", "UFL'", 'URF', 'UBR', "ULB'", "UBR'", 'ULB'),
                ("DRB'", 'DBL', 'DRB', 'DFR', 'DFL', 'DBL', "DFR'", "DBL'", "DFR'", "DFL'"),
                ("UBR'","ULB'","UBR'","ULB","UFL","URF'","UFL'","URF","UBR'"),
                ("DRB'", 'DBL', 'DRB', "DBL'", "DFR'", "DRB'", 'DFR', 'DRB', 'DBL', "DRB'", "DBL'", "DFR'", 'DRB', 'DFR'),
                ('DRB', 'UBR', "ULB'", 'DRB', "UBR'", "ULB'", "UBR'", "ULB'", 'UBR'),
                ('UBR', "ULB'", 'UBR', "ULB'", "UBR'", 'ULB', "UBR'", 'ULB'),
                ("DRB'", "DBL'", "ULB'", 'UFL', "URF'", "ULB'", "UBR'", 'ULB', "UBR'", "ULB'", "UFL'", 'ULB'),
                ("URF'", 'DFL', "DFR'", "DFL'", 'URF', "UBR'", 'UFL', "URF'", 'UFL', 'URF', 'UFL', "URF'", "UFL'", 'URF', "ULB'", 'UBR', 'ULB', "UBR'"),
                ("URF'", 'DFL', "DFR'", "DRB'", "DFL'", "ULB'", 'URF', "UBR'", "UFL'", "ULB'"),

            ],
            [
            ],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    if puzzle_type == "pyraminx":
      return (
            [
                ("R", "U", "R'", "U'"),
                ("L'", "U'", "L", "U"),
                ("R", "L'", "R'", "L"),
                ("U", "R", "U'", "R'"),
                ('U', 'R','U', "R'", 'U', 'R', 'U', "R'","u'"),
                ('L', 'R', 'U', "R'", "U'", "L'"),
                ('R', "L'", 'U', 'L', "U'", "R'"),
                ("R'","L","R","L'","U","L'","U'","L"),
            ],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    if puzzle_type == "master_pyraminx":
      return (
            [
                ("3L","3R","3L'","3R'"),
                ("3L","3R'","3L'","3R"),
                ('3R', '3B', 'L', '3B', "3U'", "R'", "3L'", "3B'", "3R'", 'L', "3U'", "3B'", "3L'", "U'", '3R'),



            ],
            [
                ("u",),
                ("l",),
                ("R","3U","R'","3U'"),
                
            ],
            [

            ],
            [],
            [],
            [],
            [],
            [],
        )


    if puzzle_type == "megaminx":
        return (
            [
                ('F', "R'", "F'", 'R', 'U', 'R', "U'", "R'"),
                ("R","U'","R'","U","R'","F","R","F'"),
                ('R', "U'", "bL'", 'U', "R'", "F'", 'R', "U'", 'bL', 'U', "R'", 'F'),
                ("B'", 'D', 'dL', "D'", 'B', 'sR', "B'", 'D', "dL'", "D'", 'B', "sR'"),
                ("U2'", 'F', 'dR', "F'", "U2'", 'L', "U'", 'F', "dR'", "F'", 'U', "L'", "U'"),
                ("bR'", 'B', 'sL', "B'", 'bR', 'sR', "bR'", 'B', "sL'", "B'", 'bR', "sR'"),
                ("bR2'", 'B', 'sL', "B'", "bR2'", 'sR', "bR'", 'B', "sL'", "B'", 'bR', "sR'", "bR'"),
                ('sL2', "L'", "F'", 'L', "sL2'", "L'", 'F', 'L', 'sL', 'D', "dR'", "D2'", "sL'", 'L', 'sL', 'D2', 'dR', "D'", "sL'", "L'"),
                ("F'", "L'", "bR", "U2", "bL", "U'", "bL'", "U'", "bR'", "bL", "L", "U", "L'", "U'", "bL'", "L", "F"),
                ("L'", "bR", "U2", "bL", "U'", "bL'", "U'", "bR'", "bL", "L", "U", "L'", "U'", "bL'", "L"),
                ('F2', "dL2'", "L2'", 'dL2', 'L', "dL'", "L'", "dL'", 'L2', 'bL', 'sL', 'dL', "sL'", 'dL', "bL'", "F2'"),
                ("bR2'", 'bL2', "sL2'", 'bL2', 'sL', "bL'", "sL'", "bL'", 'sL2', 'dL', 'L', 'bL', "L'", 'bL2', "dL'", 'bR2'),
                ("U","R'","U'","F","U","R","U'","R'","F'","R"),
                ("U'","R'","U'","F","U","R","U'","R'","F'","R"),
                ("U2","R'","U'","F","U","R","U'","R'","F'","R"),
                ("U2'","R'","U'","F","U","R","U'","R'","F'","R"),
                ("U","R2","U2","R2'","U","R2","U2","R2'","U2"),
                ("U'","R2","U2","R2'","U","R2","U2","R2'","U2"),
                ("U2","R2","U2","R2'","U","R2","U2","R2'","U2"),
                ("U2'","R2","U2","R2'","U","R2","U2","R2'","U2"),
                ('R2', "U2'", "bR2'", 'U2', 'bR', "U'", "bR'", "U'", 'bR2', 'B', 'bL', 'U', "bL'", 'U', "B'", "R2'", "F2'", 'U2', 'L2', "U2'", "L'", 'U', 'L', 'U', "L2'", "sL'", "bL'", "U'", 'bL', "U'", 'sL', 'F2'),
                ('U', 'L', 'bL', "L'", "bL'", "U'", "sL'", "bL'", "L'", 'bL', 'L', 'sL'),
                ('B', 'sL', 'D', "sL'", "D'", "B'", 'bL', 'sL', 'B', 'sL', "B'", "sL2'", "bL'"),
                ('sR', 'R', 'bR', "R'", "bR'", "sR'", "U'", "bR'", "R'", 'bR', 'R', 'U', 'B2', "D2'", "B'", 'D', 'B', 'D', "B2'", "bL'", "sL'", "D'", 'sL', 'D', 'bL'),
                ("R","U","R'","U'"),
                ("R","U'","R'","U"),
                ('D2', "B2'", 'bR2', "B2'", "bR'", 'B', 'bR', 'B', "bR2'", "U'", "bL'", "B'", 'bL', "B2'", 'U', "D2'", 'L', 'dL2', 'sL', "dL'", "sL'", "dL'", "L'", 'sL', 'D', 'dL', "D'", "dL'", "sL'", 'F2', "R2'", "dR2'", 'R2', 'dR', "R'", "dR'", "R'", 'dR2', 'D', 'sR', 'R', "sR'", 'R', "D'", "F2'"),
                ("R'", "bR'", "U'", "bR'", 'U', 'bR2', 'R', "U'", "bR'", "bL'", 'bR', 'bL', 'U'),
                ("D'", "dR'", "R'", "sR'", "bR'", 'sR', 'bR', 'R', "dR'", "sR'", "R'", "sR'", 'R', 'sR2', 'dR2', 'D')


            ],
            [
                ("R'", "bR'", "U'", "bR'", 'U', 'bR2', 'R', "U'", "bR'", "bL'", 'bR', 'bL', 'U'),
            ],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    if puzzle_type == "fto":
        return (
            [   
                ("URF",),
                ("URF'",),
                ("UFL",),
                ("UFL'",),
                ("mUFL'","UBR","UFL'","UBR'","mUFL","UBR","UFL","UBR'"),
                ("mUBR'", "DFR'", 'UBR', 'DFR', 'mUBR', "DFR'", "UBR'", 'DFR'),
                ("URF","ULB","URF'","ULB","URF","ULB","URF'","ULB"),
                ("URF","UFL","URF'","UFL'"),
                ("URF","UBR","URF'","UBR'"),


                          
            ],
            [
            ],
            [],
            [],
            [],
            [],
            [],
            [],
        )

    if puzzle_type == "cto":
        return (
            [
                ("U", "R", "U'", "R'"),
                ("U", "R'", "U'", "R"),
                ("R", "U", "R'", "U", "R", "U2", "R'"),
                ('U', "R'", 'U2', 'R', 'U'),
                ('U2', "R'", "U'", 'R', "U'"),
                ("U'", 'R2', "U'", 'R2', 'U2'),
                ('R', "U'", "R'", 'u', "U'", 'R', "U'", "R'", 'U2'),
                ("F","R","U","R'","U'","F'"),
                ("R'","F","R","F'","U","F'","U'","F"),
                ('U', 'R', 'U', "R'", 'U2'),
                ('F', "U'", 'R', 'U', "R'", "F'"),
                ("u'", 'R2', 'U', 'R2', 'U2', 'R2', 'U', 'R2', 'U'),
                ('U2', 'R', 'U2', "R'", 'U', 'R', 'U2', "R'", 'U2', 'u2', 'R', 'U', "R'"),
                ("u'", 'R', 'U', "R'", 'U', 'R', 'U', "R'", 'U2'),
                ('U2', 'R', "U'", "R'", 'U', 'R', 'U', "R'", 'U', 'R', "U'", "R'", 'u'),
                ("R'", 'U', 'R', "U'", 'R', 'U', 'R', "U'", "R'", "r'"),
                ("U'", "R'", 'U', "R'", "U'", "R'", 'U', "R'", "U'", "R'", 'U', 'r'),
                ('R2', "U'", 'R2', 'u', "U'", 'R2', "U'", 'R2', 'U2'),
                ('F', "L'", 'F', 'L', "F'", 'D', "F'", "L'", "D'", 'L', "D'", 'B', 'D', "B'"),
                ('D', 'B', 'D', "B'", 'D2', 'B', 'D', "B'", "d'", 'F', 'D', 'F', "D'", 'F', 'D', 'F', "D'", 'F', "f'"),

                
            ],
            [
            ],
            [],
            [],
            [],
            [],
            [],
            [],
        )
            
    if size == 3:
        return (
            [
                (" R "," U "," R'"," U'"," F'"," U "," F "),
                (' U ', ' R2', " U'", ' B2', ' U ', ' B2', ' U ', ' R2', " U'", ' B2', " U'", ' B2'),
                (' B ', " U'", " B'", ' U ', " B'", ' R2', ' F ', " D'", " F'", ' R2', ' L ', ' B ', ' R ', " B'", " L'", ' B ', " R'"),
                (" F'", " U ", " F ", " U ", " R ", " U'", " R'"),
                (" S "," E "," S'"," E'"),
                (' F ', " R'", " F'", ' R2', " U'", " R'", " F'", " U'", ' F ', ' R ', ' U ', " R'"),
                (" M "," U "," M2"," U2"," M2"," U "," M'"),
                (' B2', " L'", ' B2', ' D2', " F'", " R'", ' F ', ' D2', " B'", ' L ', ' B '),
                (' F2', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F ', " L'", ' F ', ' R ', " F'", ' L '),
                (' B2', ' U2', ' B ', ' U2', " B'", ' U2', ' B2', ' L2', " F'", " L'", ' F ', " L'", ' U2', ' R ', " B'", " R'", ' U2'),
                (" L "," R'",' F ', " R'", " F'", ' R2', " U'", " R'", " F'", " U'", ' F ', ' R ', ' U ', " L'"),
                (" U2"," R "," U "," R'"," U'"," F'"," U "," F "," U2"),
                (" U2"," F'", " U ", " F ", " U ", " R ", " U'", " R'"," U2"),
                (" F'", ' U2', ' F ', ' R ', ' U ', " R'", " U'", " F'", " U'", ' F '),
                (" U'", " F'", ' U2', ' F ', ' R ', ' U ', " R'", " U'", " F'", " U'", ' F ', " U "),
                (" U "," M "," R "," F "," D'"," R2"," U'"," F'"," D2"," B'"," R'"," F "," L'"),
                (" R "," U2"," D'"," S "," U'"," F "," R "," L "," D'"," R "," B'"," F2"," U2"),
                (" L "," R "," U2"," L'"," R'"),
                (" F "," B "," U2"," F'"," B'"),
                (" R "," U ") * 7,
                (" F "," U ") * 7,
                (" R "," U'") * 7,
                (" F "," U'") * 7,
                (' F2', " U'", " F'", ' U ', ' F ', ' R ', " U'", " R'", " F'", ' L ', " F'", " L'"),
                (' B2', " L'", ' B2', ' D2', " F'", " R'", ' F ', ' D2', " B'", ' L ', ' B '),
                (' F2', " R'", ' F ', ' D2', " B'", ' L ', ' B ', ' D2', ' F ', " L'", ' F ', ' R ', " F'", ' L '),
                (' B2', ' U2', ' B ', ' U2', " B'", ' U2', ' B2', ' L2', " F'", " L'", ' F ', " L'", ' U2', ' R ', " B'", " R'", ' U2'),
                (' L2', " U'", ' L ', " U'", ' F2', ' D ', " R'", " D'", ' F2', ' U2', ' L '),
                (' B ', " L'", " F'", ' L ', " B'", ' L2', ' F ', " L'", " F'", ' L2', ' F '),
                (" D'", ' L ', " U'", " L'", ' D ', ' U ', ' L2', " U'", " L'", ' U ', ' L2'),

            ],
            [
            ],
            [],
            [],
            [],
            [],
            [],
            [],
            )
    elif size == 4:
        return (
            [
                (" x2"," y "),
                ("2U "," R "," U "," R'","2U'","2R "," U "," F "," U'"," F'","2R'","2D2"," F "," U2"," F'","2D2"," F ","2R2","2U ","2R2","2U'"," F'"),
                (" R ","2U "," F ","2U'"," R'"," F'"),
                (' B2', " L'", ' B2', ' D2', " F'", " R'", ' F ', ' D2', " B'", ' L ', ' B ') + ("2F2"," R2"," U2","2F2"," U2"," R2","2F2") + ("2U ","2R "," F ","2U'","2R'"," F'"),
                (' B2', " L'", ' B2', ' D2', " F'", " R'", ' F ', ' D2', " B'", ' L ', ' B ') + ("2F2"," R2"," U2","2F2"," U2"," R2","2F2") + ("2F "," R'","2F'","2U'"," R ","2U "),
            ],
            [
            ],
            [
            ],
            [

            ],
            [

            ],
            [],
            [],
            [],
        )


    elif size == 7:
        return (
            [(" U "," R "),
             (" U'"," F'"),
             (" R'"," F "," R "," F'"),
             (" F "," R'"," F'"," R "),
             (" F "," E "," F "," E "," F'"," E "," F "," E "," F'"," E "," F'"),
             ("2U ","3R "),
             ("2U'","3F'"),
             ("2R ","2L ","2F ","2B "),
             ("2F'","2B'","2R'","2L'"),
             (" y "," x2"),
             (" y "," z2"),
             ("2L2"," U ","3F2","3R2"," U'"),
             ('2U2', " B'", '2R2', ' B ', ' R2', " B'", '2R2', " B'", "2U'", ' B2', '2U2', ' R2', "2D'", ' F2', '2U2', ' F2', ' R2', '2D ', ' R2'),
             ('3U2', " B'", '3R2', ' B ', ' R2', " B'", '3R2', " B'", "3U'", ' B2', '3U2', ' R2', "3D'", ' F2', '3U2', ' F2', ' R2', '3D ', ' R2'),
             (" M "," U "," M2"," U2"," M2"," U "," M'"),


            ],
            [
            ],
            [
            ],
            [
            ],
            [],
            [],
            [],
            [],
        )


def build_default_frame_config():
    """現在の既定実験設定を FrameConfig として返す。"""
    ai_search_modes = [
        'search2'
        if ai_index < 10
        else 'search2'
        for ai_index in range(20)
    ]
    original_transformer_attention = [False] * 10 + [True] * 10
    ai_count = len(ai_search_modes)
    is_search2_ai = [mode.startswith('search2') for mode in ai_search_modes]
    lrs = [
        1.0e-5 if original_transformer_attention[ai_index] else (2.0e-6 if is_search2_ai[ai_index] else 1.0e-6)
        for ai_index in range(ai_count)
    ]
    wdlrs = [
        1.0e-7 if original_transformer_attention[ai_index] else (1.0e-7 if is_search2_ai[ai_index] else 1.0e-5)
        for ai_index in range(ai_count)
    ]
    skip_search = [is_search2_ai[ai_index] for ai_index in range(ai_count)]
    weight_decay = [True] * ai_count
    search3_progress = [False] * 10 + [False,False,False,True,False,True,False,True,False,False]
    residuals = [True] * ai_count
    #search2_value_loss_types = ['myloss','myloss','myloss2','myloss2','myloss2','myloss2','myloss2','myloss2','myloss','myloss'] * 2
    search2_value_loss_types = ['myloss','myloss','myloss','myloss','myloss','myloss','myloss','myloss','myloss','myloss'] * 2


    adam = weight_decay.copy()

    cube_size = 4
    puzzle_type = 'cube'
    if cube_size >= 6:
        transform_idx = [0,49,50,3,52,5,54,7,24,25] * 2
        flip_inside_idx = [False,True] * 10
    else:
        transform_idx = [0,1,2,3,4,5,6,7,24,25] * 2
        flip_inside_idx = [False] * ai_count


    if puzzle_type == 'megaminx':
        priority_list = [['Corner', 'MidEdge']] * ai_count
        bootstrap_datas = None
        bootstrap_search3_datas = None
        transform_idx = [0,1,2,3,4,5,6,7,8,9] * 2
        flip_inside_idx = [False] * ai_count
    elif puzzle_type == 'pyraminx':
        priority_list = [['Corner', 'Edge', 'Center']] * ai_count
        bootstrap_datas = None
        bootstrap_search3_datas = None
        transform_idx = [0] * ai_count
        flip_inside_idx = [False] * ai_count
    elif puzzle_type == 'master_pyraminx':
        priority_list = [['Corner', 'Edge', 'MidEdge', 'Center']] * ai_count
        bootstrap_datas = None
        bootstrap_search3_datas = None
        transform_idx = [0] * ai_count
        flip_inside_idx = [False] * ai_count
    elif puzzle_type == 'skewb':
        priority_list = [['Corner', 'Center']] * ai_count
        bootstrap_datas = None
        bootstrap_search3_datas = None
        transform_idx = [0] * ai_count
        flip_inside_idx = [False] * ai_count
    elif puzzle_type == 'square1':
        priority_list = [['Corner', 'Edge', 'Shape']] * ai_count
        bootstrap_datas = None
        bootstrap_search3_datas = None
        transform_idx = [0] * ai_count
        flip_inside_idx = [False] * ai_count
    elif puzzle_type == 'fto':
        priority_list = [['Corner', 'Edge', 'CenterA', 'CenterB']] * ai_count
        bootstrap_datas = None
        bootstrap_search3_datas = None
        transform_idx = [0] * ai_count
        flip_inside_idx = [False] * ai_count
    elif puzzle_type == 'cto':
        priority_list = [['Corner', 'Edge', 'Center']] * ai_count
        bootstrap_datas = None
        bootstrap_search3_datas = None
        transform_idx = [0] * ai_count
        flip_inside_idx = [False] * ai_count
    else:
        priority_list = [
            ['CoreCenter','ObliqueCenter-A','PlusCenter-Layer2','XCenter-Layer2','ObliqueCenter-B','PlusCenter-Layer3','XCenter-Layer3','Wing-Layer2','Wing-Layer3','Corner','MidEdge'],
            ['Wing-Layer3','Wing-Layer2','MidEdge','Corner','XCenter-Layer2','PlusCenter-Layer2','ObliqueCenter-A','XCenter-Layer3','PlusCenter-Layer3','ObliqueCenter-B','CoreCenter'],
        ] * 10
        bootstrap_datas = build_default_bootstrap_datas(cube_size = cube_size)
        bootstrap_search3_datas = None

    return FrameConfig(
        puzzle_type = puzzle_type,
        cube_size = cube_size,
        ai_search_modes = ai_search_modes,
        initial_scramble_groups = _default_initial_scramble_groups(cube_size,puzzle_type),
        transform_random = False,
        search3_progress = search3_progress,
        lrs = lrs,
        wdlrs = wdlrs,
        skip_search = skip_search,
        weight_decay = weight_decay,
        adam = adam,
        lr_vs = [0.99] * ai_count,
        lr_hs = [0.99] * ai_count,
        out_cs = [1.0] * ai_count,
        search3_cs = [0.05] * ai_count,
        search2_max_frontiers = [30000] * ai_count,
        search2_torch_batch_sizes = [
            64 if original_transformer_attention[ai_index] else 100
            for ai_index in range(ai_count)
        ],
        search2_value_loss_types = search2_value_loss_types,
        torch_training_devices = [
            'cpu' if original_transformer_attention[ai_index] else 'auto'
            for ai_index in range(ai_count)
        ],
        use_torch = [False] * ai_count,
        use_torch_predict = [
            bool(original_transformer_attention[ai_index])
            for ai_index in range(ai_count)
        ],
        use_torch_training = [
            bool(original_transformer_attention[ai_index])
            for ai_index in range(ai_count)
        ],
        residuals = residuals,
        update_scales = [
            (5.0, 1.0, 20.0) if is_search2_ai[ai_index] else (5.0, 1.0, 5.0)
            for ai_index in range(ai_count)
        ],
        original_transformer_attention = original_transformer_attention,
        original_transformer_attention_dims = [64] * ai_count,
        original_transformer_attention_token_modes = ['piece'] * ai_count,
        original_piece_attention_backward_chunk_sizes = [32] * ai_count,
        original_train_batch_sizes = [
            20 if original_transformer_attention[ai_index] else 100
            for ai_index in range(ai_count)
        ],
        original_train_state_batch_sizes = [
            16 if original_transformer_attention[ai_index] else 0
            for ai_index in range(ai_count)
        ],
        original_train_max_batches = [
            100 if original_transformer_attention[ai_index] else 0
            for ai_index in range(ai_count)
        ],
        original_train_recent_ratios = [
            1.0 if original_transformer_attention[ai_index] else 0.0
            for ai_index in range(ai_count)
        ],
        max_search2_data = 80000,
        max_search3_data_per_ai = 2000,
        transform_idx = transform_idx,
        flip_inside_idx = flip_inside_idx,
        priority_list = priority_list,
        bootstrap_datas = bootstrap_datas,
        bootstrap_search3_datas = bootstrap_search3_datas,
    )


if __name__ == '__main__':
    config = build_default_frame_config()
    F = Frame(config = config)
    F.pack()
    F.mainloop()
