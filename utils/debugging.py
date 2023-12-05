import random
import torch
import numpy as np

from collections import namedtuple
from environments.bounding_boxes import get_bounding_box
from linear_solvers import init_linear_solver
from oracles import init_oracle
from outer_loops import init_outer_loop

DummyEnv = namedtuple('DummyEnv', ['env_id'])


def debug_experiment(method, algorithm, config, outer_params, wandb_summary, callback=None):
    env_id = config['env_id']
    num_objectives = config['num_objectives']
    seed = config['seed']
    wandb_project_name = config['wandb_project_name']
    wandb_entity = config['wandb_entity']
    run_name = f'{method}__{algorithm}__{env_id}__{seed}'

    # Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Setup environment.
    minimals, maximals, ref_point = get_bounding_box(env_id)
    env = DummyEnv(env_id)

    linear_solver = init_linear_solver('known_box', minimals=minimals, maximals=maximals)
    oracle = init_oracle(algorithm,
                         wandb_summary)
    ol = init_outer_loop(method,
                         env,
                         num_objectives,
                         oracle,
                         linear_solver,
                         ref_point=ref_point,
                         exp_name=run_name,
                         wandb_project_name=wandb_project_name,
                         wandb_entity=wandb_entity,
                         seed=seed,
                         **outer_params)
    ol.solve(callback=callback)
    return ol.hv


if __name__ == '__main__':
    wandb_summary = {
        "ideal_19": [
            41,
            41,
            41,
            41
        ],
        "referent_9": [
            -50,
            17.156372353578917,
            -51,
            -51
        ],
        "pareto_point_11": [
            18.334306188225742,
            9.328281966354698,
            15.971360861249268,
            11.351779298111795
        ],
        "pareto_point_50": [
            17.037746839821338,
            15.76179650751874,
            18.6410554240644,
            14.883366599716249
        ],
        "ideal_59": [
            41,
            41,
            41,
            41
        ],
        "referent_41": [
            -51,
            -51,
            19.52676356639713,
            10.079772056024522
        ],
        "pareto_point_61": [
            18.457509709512596,
            8.872949840447399,
            11.268995372533643,
            15.483429623935372
        ],
        "referent_4": [
            -50,
            -51,
            -51,
            14.81131527364254
        ],
        "referent_50": [
            15.889073267681525,
            15.969543724227696,
            -51,
            -51
        ],
        "referent_52": [
            -51,
            -50,
            -51,
            15.013395869079975
        ],
        "pareto_point_13": [
            16.154220285490155,
            13.566378804366105,
            16.620134588107465,
            13.741893598635215
        ],
        "referent_34": [
            -51,
            -51,
            21.581617323458193,
            -50
        ],
        "pareto_point_42": [
            16.85894548599492,
            14.621479918489932,
            18.56195450179912,
            13.48604561826709
        ],
        "_wandb.runtime": 1162,
        "ideal_56": [
            41,
            41,
            41,
            41
        ],
        "referent_8": [
            -51,
            -50,
            19.52676356639713,
            -51
        ],
        "referent_16": [
            18.399074690192936,
            -50,
            -51,
            -51
        ],
        "referent_31": [
            18.399074690192936,
            -51,
            -51,
            -50
        ],
        "pareto_point_5": [
            18.406302155852316,
            14.805061578536408,
            18.000446898788212,
            15.01331633547321
        ],
        "pareto_point_58": [
            15.987186759859323,
            12.912265811190007,
            14.553995039872826,
            13.867090774229728
        ],
        "ideal_42": [
            41,
            41,
            41,
            41
        ],
        "referent_47": [
            -51,
            -51,
            19.029450749009847,
            11.225904784239829
        ],
        "pareto_point_36": [
            17.094730689674616,
            14.84029560763389,
            19.64954813361168,
            13.663358521629124
        ],
        "ideal_16": [
            41,
            41,
            41,
            41
        ],
        "ideal_31": [
            41,
            41,
            41,
            41
        ],
        "ideal_33": [
            41,
            41,
            41,
            41
        ],
        "ideal_10": [
            41,
            41,
            41,
            41
        ],
        "referent_48": [
            15.889073267681525,
            15.81954015865922,
            -51,
            -51
        ],
        "pareto_point_45": [
            17.56542617470026,
            8.90113926500082,
            20.32738587245345,
            7.142719865832478
        ],
        "pareto_point_62": [
            15.25984844959341,
            14.609999755276368,
            18.750014875810592,
            12.076977056544274
        ],
        "ideal_54": [
            41,
            41,
            41,
            41
        ],
        "ideal_57": [
            41,
            41,
            41,
            41
        ],
        "ideal_46": [
            41,
            41,
            41,
            41
        ],
        "referent_49": [
            -51,
            -51,
            19.191632559746505,
            11.225904784239829
        ],
        "referent_60": [
            17.005482772141693,
            -51,
            17.99347320869565,
            10.079772056024522
        ],
        "pareto_point_7": [
            18.380597175210713,
            14.776565758017822,
            17.997327811419964,
            15.013355624135585
        ],
        "pareto_point_12": [
            12.806558504039424,
            10.83558818657446,
            15.944297095032034,
            8.757614472929527
        ],
        "ideal_4": [
            41,
            41,
            41,
            41
        ],
        "ideal_29": [
            41,
            41,
            41,
            41
        ],
        "ideal_55": [
            41,
            41,
            41,
            41
        ],
        "referent_3": [
            -50,
            -51,
            -51,
            14.473640073053538
        ],
        "referent_13": [
            -51,
            -51,
            19.52676356639713,
            -50
        ],
        "pareto_point_15": [
            18.115608953000045,
            3.306519956546035,
            17.557148625447006,
            3.8856730494089424
        ],
        "pareto_point_27": [
            16.19027279790491,
            16.54745766149834,
            18.19660727005452,
            15.25272557914257
        ],
        "pareto_point_32": [
            14.065504155289382,
            15.315819950574078,
            20.489850856401027,
            10.737172620987986
        ],
        "pareto_point_46": [
            15.931006412878633,
            13.685523568093776,
            17.654268018230795,
            12.825211276807822
        ],
        "ideal_32": [
            41,
            41,
            41,
            41
        ],
        "ideal_58": [
            41,
            41,
            41,
            41
        ],
        "pareto_point_9": [
            18.380466844439507,
            14.791251196535304,
            17.99325647905469,
            15.0132933267951
        ],
        "pareto_point_28": [
            11.825734003940598,
            16.044325830220938,
            14.75109559900593,
            14.05994178547524
        ],
        "ideal_1": [
            41,
            41,
            41,
            41
        ],
        "pareto_point_26": [
            14.773837880045177,
            13.922863806681708,
            18.044616179957075,
            11.532822661064564
        ],
        "ideal_39": [
            41,
            41,
            41,
            41
        ],
        "referent_24": [
            -51,
            11.93015021827945,
            19.52676356639713,
            -51
        ],
        "referent_35": [
            -50,
            -51,
            -51,
            15.013395869079975
        ],
        "referent_58": [
            -51,
            -51,
            17.99347320869565,
            14.81131527364254
        ],
        "ideal_9": [
            41,
            41,
            41,
            41
        ],
        "pareto_point_37": [
            16.256999117587693,
            13.21089854439546,
            17.976975679844617,
            11.91524255083874
        ],
        "ideal_48": [
            41,
            41,
            41,
            41
        ],
        "referent_22": [
            -50,
            -51,
            21.581617323458193,
            -51
        ],
        "pareto_point_47": [
            15.321163277477028,
            17.19274891143665,
            17.882851757016034,
            15.248592402487994
        ],
        "pareto_point_59": [
            16.50693296864629,
            15.520850504077972,
            19.103274027109148,
            13.81162548283115
        ],
        "referent_37": [
            19.508549706861377,
            -51,
            -50,
            -51
        ],
        "pareto_point_17": [
            18.15712125584483,
            11.93015021827945,
            21.581617323458193,
            10.079772056024522
        ],
        "ideal_53": [
            41,
            41,
            41,
            41
        ],
        "pareto_point_24": [
            14.103505857847631,
            14.554519309028985,
            13.168045666515829,
            16.9911288921535
        ],
        "referent_23": [
            -51,
            -51,
            18.751511723846196,
            11.225904784239829
        ],
        "ideal_61": [
            41,
            41,
            41,
            41
        ],
        "referent_57": [
            -51,
            14.821070071309803,
            -51,
            14.81131527364254
        ],
        "pareto_point_38": [
            19.10227130487561,
            8.479803885072469,
            15.195635791644454,
            10.103358885291964
        ],
        "ideal_44": [
            41,
            41,
            41,
            41
        ],
        "referent_20": [
            -51,
            17.156372353578917,
            -50,
            -51
        ],
        "ideal_26": [
            41,
            41,
            41,
            41
        ],
        "referent_2": [
            -51,
            -51,
            -50,
            11.225904784239829
        ],
        "referent_54": [
            -51,
            -51,
            18.751511723846196,
            14.32524123667972
        ],
        "pareto_point_30": [
            15.022436377555133,
            16.030671387221663,
            19.868056415244936,
            12.238851996613668
        ],
        "ideal_6": [
            41,
            41,
            41,
            41
        ],
        "ideal_8": [
            41,
            41,
            41,
            41
        ],
        "ideal_35": [
            41,
            41,
            41,
            41
        ],
        "referent_43": [
            18.15712125584483,
            -51,
            17.99347320869565,
            -51
        ],
        "referent_61": [
            13.323648690674451,
            11.93015021827945,
            19.191632559746505,
            -51
        ],
        "ideal_52": [
            41,
            41,
            41,
            41
        ],
        "referent_33": [
            -50,
            17.156372353578917,
            -51,
            -51
        ],
        "pareto_point_29": [
            16.534124350982644,
            14.079808008314576,
            17.886448874788766,
            13.31220402962994
        ],
        "ideal_18": [
            41,
            41,
            41,
            41
        ],
        "ideal_24": [
            41,
            41,
            41,
            41
        ],
        "referent_17": [
            -50,
            -51,
            19.52676356639713,
            -51
        ],
        "referent_29": [
            18.15712125584483,
            -51,
            17.99347320869565,
            -51
        ],
        "referent_40": [
            -51,
            11.93015021827945,
            19.52676356639713,
            -51
        ],
        "ideal_7": [
            41,
            41,
            41,
            41
        ],
        "ideal_43": [
            41,
            41,
            41,
            41
        ],
        "referent_26": [
            -51,
            15.81954015865922,
            -51,
            11.225904784239829
        ],
        "referent_38": [
            -50,
            -51,
            21.581617323458193,
            -51
        ],
        "referent_46": [
            17.005482772141693,
            14.821070071309803,
            -51,
            -51
        ],
        "pareto_point_34": [
            18.018586500305393,
            13.289584134817124,
            16.463201721117365,
            14.207362213824762
        ],
        "referent_10": [
            -51,
            -51,
            -50,
            15.013344669546932
        ],
        "pareto_point_40": [
            13.32406489020731,
            14.02576301517358,
            13.28728972116951,
            14.059908352484928
        ],
        "ideal_30": [
            41,
            41,
            41,
            41
        ],
        "ideal_45": [
            41,
            41,
            41,
            41
        ],
        "outer/coverage": 0.999991411393109,
        "pareto_point_8": [
            18.38721027269959,
            14.79151793017052,
            18.019496403336525,
            15.013373686168343
        ],
        "ideal_11": [
            41,
            41,
            41,
            41
        ],
        "ideal_37": [
            41,
            41,
            41,
            41
        ],
        "referent_39": [
            -51,
            -51,
            18.751511723846196,
            11.225904784239829
        ],
        "pareto_point_3": [
            17.005482772141693,
            15.81954015865922,
            18.751511723846196,
            14.81131527364254
        ],
        "pareto_point_19": [
            17.64192802399397,
            15.052922242041678,
            19.89286133274436,
            13.415753256031312
        ],
        "ideal_12": [
            41,
            41,
            41,
            41
        ],
        "ideal_34": [
            41,
            41,
            41,
            41
        ],
        "pareto_point_14": [
            13.079162808601016,
            9.23969533205389,
            10.89930266498908,
            10.573625264851437
        ],
        "pareto_point_23": [
            15.976936619502958,
            14.465994826499372,
            17.921323065647083,
            13.330582451047375
        ],
        "referent_25": [
            13.323648690674451,
            15.81954015865922,
            -51,
            -51
        ],
        "pareto_point_18": [
            17.489524035304786,
            15.325722486567685,
            18.97115612015128,
            14.501904420547651
        ],
        "pareto_point_44": [
            17.122793325036763,
            15.662505280394107,
            18.694324048608543,
            14.85563690252602
        ],
        "ideal_5": [
            41,
            41,
            41,
            41
        ],
        "ideal_38": [
            41,
            41,
            41,
            41
        ],
        "pareto_point_4": [
            18.391154274344444,
            14.821070071309803,
            17.98530258759856,
            15.013344669546932
        ],
        "ideal_15": [
            41,
            41,
            41,
            41
        ],
        "ideal_14": [
            41,
            41,
            41,
            41
        ],
        "ideal_25": [
            41,
            41,
            41,
            41
        ],
        "ideal_40": [
            41,
            41,
            41,
            41
        ],
        "referent_1": [
            -51,
            -51,
            -50,
            -50
        ],
        "referent_44": [
            18.399074690192936,
            -51,
            -51,
            11.513224198129029
        ],
        "referent_51": [
            -51,
            15.969543724227696,
            -51,
            14.016914363838731
        ],
        "pareto_point_20": [
            18.229725954309107,
            13.392366741197876,
            15.72337621666491,
            15.29174904008396
        ],
        "pareto_point_57": [
            15.808920186385512,
            16.23315853362903,
            20.004324076026677,
            13.20903223855421
        ],
        "ideal_22": [
            41,
            41,
            41,
            41
        ],
        "ideal_28": [
            41,
            41,
            41,
            41
        ],
        "iteration": 63,
        "_timestamp": 1701767742.337701,
        "pareto_point_33": [
            16.562029124349355,
            16.0849091851525,
            19.42304197266698,
            14.093613994177431
        ],
        "pareto_point_54": [
            18.21045212864876,
            13.923098072651774,
            18.39358593016863,
            13.611544608795084
        ],
        "ideal_17": [
            41,
            41,
            41,
            41
        ],
        "referent_12": [
            -50,
            17.156372353578917,
            -51,
            -51
        ],
        "pareto_point_2": [
            16.9388227827847,
            15.471592666041106,
            18.69049440033734,
            14.473640073053538
        ],
        "pareto_point_25": [
            16.87456900767982,
            13.782738457429224,
            16.712175748232983,
            15.255558937452731
        ],
        "pareto_point_48": [
            16.726896665096284,
            15.969543724227696,
            19.191632559746505,
            14.32524123667972
        ],
        "referent_21": [
            18.399074690192936,
            -50,
            -51,
            -51
        ],
        "pareto_point_39": [
            15.889073267681525,
            16.354426338840277,
            19.029450749009847,
            14.016914363838731
        ],
        "outer/discarded_hv": 48842725.90722032,
        "referent_36": [
            -51,
            17.156372353578917,
            -50,
            -51
        ],
        "referent_32": [
            -51,
            -51,
            -50,
            15.013395869079975
        ],
        "ideal_41": [
            41,
            41,
            41,
            41
        ],
        "pareto_point_43": [
            12.038596594746632,
            17.300896609092415,
            17.153719671925064,
            11.54614995833719
        ],
        "pareto_point_6": [
            18.40227224752307,
            14.839344471227378,
            17.99564435765147,
            15.013306914716958
        ],
        "ideal_21": [
            41,
            41,
            41,
            41
        ],
        "ideal_62": [
            41,
            41,
            41,
            41
        ],
        "referent_19": [
            -50,
            -51,
            -51,
            15.013395869079975
        ],
        "referent_27": [
            -51,
            -51,
            19.52676356639713,
            10.079772056024522
        ],
        "ideal_36": [
            41,
            41,
            41,
            41
        ],
        "referent_6": [
            -51,
            17.156372353578917,
            -50,
            -51
        ],
        "referent_7": [
            18.391154274344444,
            -51,
            -50,
            -51
        ],
        "referent_15": [
            -51,
            17.156372353578917,
            -50,
            -51
        ],
        "referent_56": [
            -51,
            -50,
            21.581617323458193,
            -51
        ],
        "pareto_point_52": [
            15.549704546052965,
            12.600870509073138,
            20.082525984048843,
            10.287184292413292
        ],
        "pareto_point_56": [
            16.65599108055234,
            14.814878593198957,
            18.892017576545477,
            13.739207805646585
        ],
        "ideal_13": [
            41,
            41,
            41,
            41
        ],
        "ideal_51": [
            41,
            41,
            41,
            41
        ],
        "referent_14": [
            -50,
            -51,
            -51,
            15.013395869079975
        ],
        "pareto_point_31": [
            19.508549706861377,
            9.145154439430152,
            17.842763206497764,
            11.513224198129029
        ],
        "pareto_point_51": [
            17.033714416176082,
            15.76739741789177,
            18.678072195351124,
            14.880535365231337
        ],
        "ideal_50": [
            41,
            41,
            41,
            41
        ],
        "pareto_point_41": [
            15.836311221197247,
            16.86026385683566,
            18.419913289025427,
            14.897735437825322
        ],
        "outer/dominated_hv": 23313487.415980108,
        "ideal_23": [
            41,
            41,
            41,
            41
        ],
        "pareto_point_49": [
            16.69127495668421,
            15.591248719947908,
            18.46548655217557,
            14.565438275448978
        ],
        "ideal_2": [
            41,
            41,
            41,
            41
        ],
        "ideal_47": [
            41,
            41,
            41,
            41
        ],
        "referent_42": [
            13.323648690674451,
            16.354426338840277,
            -51,
            -51
        ],
        "referent_62": [
            -51,
            15.81954015865922,
            -51,
            14.32524123667972
        ],
        "pareto_point_1": [
            13.323648690674451,
            17.156372353578917,
            19.52676356639713,
            11.225904784239829
        ],
        "referent_30": [
            -51,
            -51,
            -50,
            15.013395869079975
        ],
        "pareto_point_10": [
            18.399074690192936,
            14.780880551133304,
            17.99347320869565,
            15.013395869079975
        ],
        "pareto_point_16": [
            16.82722806081176,
            15.713679835926742,
            18.486240455210208,
            14.571150197845418
        ],
        "outer/error": 1.151638483703138,
        "pareto_point_22": [
            16.815099130929447,
            15.20186226511374,
            18.326527644261805,
            14.401233741305768
        ],
        "pareto_point_35": [
            14.525565755816642,
            15.984157247403637,
            18.07118773486465,
            13.0524996894598
        ],
        "referent_28": [
            17.005482772141693,
            14.821070071309803,
            -51,
            -51
        ],
        "referent_45": [
            -51,
            16.354426338840277,
            -51,
            11.225904784239829
        ],
        "outer/hypervolume": 21995249.847214468,
        "PF_size": 25,
        "ideal_20": [
            41,
            41,
            41,
            41
        ],
        "ideal_27": [
            41,
            41,
            41,
            41
        ],
        "pareto_point_21": [
            15.012951523419469,
            12.897965024895964,
            19.990414418429136,
            9.924141663499176
        ],
        "_step": 63,
        "_runtime": 1162.869942188263,
        "pareto_point_53": [
            12.827739579262415,
            10.882261099670725,
            14.504375200485082,
            10.139891874481693
        ],
        "replay_triggered": 5,
        "referent_55": [
            19.508549706861377,
            -51,
            -51,
            -50
        ],
        "pareto_point_60": [
            16.99875646814704,
            15.783602534867825,
            18.72642029955983,
            14.81123244639486
        ],
        "ideal_3": [
            41,
            41,
            41,
            41
        ],
        "ideal_60": [
            41,
            41,
            41,
            41
        ],
        "referent_59": [
            18.399074690192936,
            -51,
            17.842763206497764,
            -51
        ],
        "ideal_49": [
            41,
            41,
            41,
            41
        ],
        "referent_11": [
            -51,
            -50,
            -51,
            15.013395869079975
        ],
        "pareto_point_55": [
            11.064828479774295,
            18.449681109879165,
            20.10915835559368,
            9.355350895174778
        ],
        "referent_5": [
            -50,
            -51,
            -51,
            15.013344669546932
        ],
        "referent_18": [
            -51,
            -51,
            21.581617323458193,
            -50
        ],
        "referent_53": [
            -51,
            17.156372353578917,
            -51,
            -50
        ]
    }
    config = {'env_id': 'mo-reacher-v4',
              'num_objectives': 4,
              'max_episode_steps': 50,
              'one_hot_wrapper': False,
              'gamma': 0.99,
              'seed': 1,
              'wandb_project_name': 'test',
              'wandb_entity': None}
    outer_params = {'tolerance': 0.00001}
    debug_experiment('IPRO', 'debug', config, outer_params, wandb_summary, callback=None)
