import requests
import os
from configparser import ConfigParser
import csv
import pickle
import pandas as pd

parser = ConfigParser()
parser.read('/home/vaibhavs/Master_Thesis/ma-vaibhav/Code/config.ini')


class DataGetter:

    def __init__(self):
        self.project_id = parser.get('general', 'project_id')
        self.base_url = parser.get('general', 'base_url')
        self.auth = (os.getenv(parser.get('auth', 'environment_variable_username')),
                     os.getenv(parser.get('auth', 'environment_variable_password')))
        self.projects = {}
        self.datapoints = []
        self.df_list = []
        self.df_data = pd.DataFrame()
        print(self.auth)
        print(self.validate_user())
        # print(self.get_project_info())

    def validate_user(self):
        r = requests.get(f"{self.base_url}/v2/user",
                         auth=self.auth)
        # r = requests.get(f"{self.base_url}/v2/user",
        #                  auth=('vaibhav.swaminathan@eonerc.rwth-aachen.de','AZTEC454#vopik'))
        return r.json()

    def get_project_info(self):
        r = requests.get(f"{self.base_url}/v2/project/{self.project_id}",
                         auth=self.auth)
        return r.json()

    def get_dps_names(self):
        r = requests.get(f"{self.base_url}/v2/project/{self.project_id}/datapoints/byPage",
                         auth=self.auth)
        dps = r.json()
        # print(dps)
        self.datapoints = [i for i in dps]
        # print(self.datapoints)

    def get_project_ids(self, only_id=True):
        r = requests.get(f"{self.base_url}/v2/user/projects", auth=self.auth)
        rj = r.json()
        for i in rj:
            if only_id:
                self.projects[i['project']['name']] = i['project']['id']
            else:
                self.projects[i['project']['name']] = i['project']
        if only_id:
            print(f'The available projects and their corresponding IDs of this user are: '
                  f'{self.projects}')
        else:
            print(f'The available projects and the corresponding data of this user are: '
                  f'{self.projects}')

    def get_gps_timeseries(self, data: list = None):
        if data is None:
            data = self.datapoints
        params = {'project_id': self.project_id,
                  'dataPointID': 'dummy',
                  'start': '2022-11-01 00:00:00',
                  'end': '2024-02-26 00:00:00',
                  'samplerate': '15m'}
        for i in data:
            params['dataPointID'] = i
            r = requests.get(f"{self.base_url}/v2/datapoint/timeseries",
                             auth=self.auth, params=params)
            rj = r.json()
            # print(rj)
            df = pd.DataFrame(rj['data']).rename({'time': 'time', 'value': i}, axis=1).set_index(
                'time')
            self.df_list.append(df)
        self.df_data = pd.concat(self.df_list,axis=1)
        # fill NaNs with previous value and sort by time index
        # this might leave the first entries with Nans, because there is no previous value. It is
        # possible to fill these with the following value, but this might cause troubles.
        self.df_data = self.df_data.sort_index().fillna(method='ffill')
        return self.df_data


if __name__ == '__main__':
    dat = DataGetter()
    # dat.get_project_ids(only_id=True)
    dat.get_dps_names()
    # vals = dat.get_gps_timeseries(data=['fAHUFlapETAActADS,fAHUFlapETASetADS,fAHUFanSUPVdpActADS,fAHUFanSUPVolFlowActADS,fAHUTempETAADS,fAHUTempSUPADS,bAHUFrostProtADS,fAHUPHValveActADS,fAHUPHValveSetADS,eAHUPHPumpContModeADS,bAHUCOPumpSetOnADS,fAHUCOPumpVolFlowADS,fAHUSHSetpointADS'])
    datapoints_trial1 = ['ADS.fAHUPHTempSupADS','ADS.fAHUPHTempRetADS','ADS.fAHUHRTempOutEstimatedADS','ADS.fAHUCOTempSupPrimADS','ADS.fAHUCOTempRetPrimADS','ADS.fAHUCOTempSupADS','ADS.fAHUCOTempRetADS','ADS.fAHURHTempSupPrimADS','ADS.fAHURHTempRetPrimADS','ADS.fAHURHTempSupADS','ADS.fAHURHTempRetADS']
    datapoints_trial2 = ['ADS.fAHUFlapEHAActADS', 'ADS.fAHUFlapEHASetADS','ADS.fAHUFlapETAActADS', 'ADS.fAHUFlapETASetADS', 'ADS.fAHUFanETASpeedActADS','ADS.fAHUFanETASpeedSetADS','ADS.fAHUFanETAVdpActADS','ADS.fAHUFanETAVolFlowActADS','ADS.fAHUFanSUPVolFlowActADS','ADS.fAHUTempEHAADS','ADS.fAHUTempETAADS','ADS.fAHUHumEHAADS','ADS.fAHUHumETAADS']
    
    datapoints_preheater = ['ADS.fAHUPHValveActADSInternalValuesMirror',
                            'ADS.fAHUPHValveSetADSInternalValuesMirror',
                            'ADS.eAHUPHPumpContModeADSInternalValuesMirror',
                            'ADS.bAHUPHPumpErrorADSInternalValuesMirror',
                            'ADS.bAHUPHPumpOperatingADSInternalValuesMirror',
                            'ADS.fAHUPHPumpPowerADSInternalValuesMirror',
                            'ADS.fAHUPHPumpPressureADSInternalValuesMirror',
                            'ADS.bAHUPHPumpSetOnADSInternalValuesMirror',
                            'ADS.fAHUPHPumpSetpointADSInternalValuesMirror',
                            'ADS.fAHUPHPumpSpeedADSInternalValuesMirror',
                            'ADS.fAHUPHPumpVolFlowADSInternalValuesMirror',
                            'ADS.fAHUPHTempSupPrimADSInternalValuesMirror',
                            'ADS.fAHUPHTempRetPrimADSInternalValuesMirror',
                            'ADS.fAHUPHTempSupADSInternalValuesMirror',
                            'ADS.fAHUPHTempRetADSInternalValuesMirror']
    
    datapoints_ETA = ['ADS.fAHUFlapETAActADSInternalValuesMirror',
                      'ADS.fAHUFlapETASetADSInternalValuesMirror',
                      'ADS.fAHUFanETASpeedActADSInternalValuesMirror',
                      'ADS.fAHUFanETASpeedSetADSInternalValuesMirror',
                      'ADS.fAHUFanETAVdpActADSInternalValuesMirror',
                      'ADS.fAHUFanETAVolFlowActADSInternalValuesMirror',
                      'ADS.fAHUTempETAADSInternalValuesMirror',
                      'ADS.fAHUHumETAADSInternalValuesMirror']
    
    datapoints_ETA_ext = ['ADS.fAHUFlapETAActADSInternalValuesMirror',
                      'ADS.fAHUFlapETASetADSInternalValuesMirror',
                      'ADS.fAHUFanETASpeedActADSInternalValuesMirror',
                      'ADS.fAHUFanETASpeedSetADSInternalValuesMirror',
                      'ADS.fAHUFanETAVdpActADSInternalValuesMirror',
                      'ADS.fAHUFanETAVolFlowActADSInternalValuesMirror',
                      'ADS.fAHUTempETAADSInternalValuesMirror',
                      'ADS.fAHUHumETAADSInternalValuesMirror',
                      'ADS.fAHUFanSUPVdpActADSInternalValuesMirror',
                      'ADS.fAHUFanSUPVolFlowActADSInternalValuesMirror']
    
    datapoints_SubSys = ['ADS.fAHUHRBypValveSetADSInternalValuesMirror',
                         'ADS.fAHUHRBypValveAct1ADSInternalValuesMirror',
                         'ADS.fAHUPHValveSetADSInternalValuesMirror',
                         'ADS.fAHUPHValveActADSInternalValuesMirror',
                         'ADS.eAHUPHPumpContModeADSInternalValuesMirror',
                         'ADS.bAHUPHPumpErrorADSInternalValuesMirror',
                         'ADS.fAHUPHPumpSpeedADSInternalValuesMirror',
                         'ADS.fAHUPHPumpVolFlowADSInternalValuesMirror',
                         'ADS.fAHUPHTempSupPrimADSInternalValuesMirror',
                         'ADS.fAHUPHTempRetPrimADSInternalValuesMirror',
                         'ADS.fAHUTempEHAADSInternalValuesMirror',
                         'ADS.fAHUHumEHAADSInternalValuesMirror',
                         'ADS.fAHUFlapEHAActADSInternalValuesMirror',
                         'ADS.fAHUFlapEHASetADSInternalValuesMirror',
                         'ADS.fAHUFlapODAActADSInternalValuesMirror',
                         'ADS.fAHUFlapODASetADSInternalValuesMirror',
                         'ADS.fAHUTempODAADSInternalValuesMirror',
                         'ADS.fAHUHumODAADSInternalValuesMirror']
    
    datapoints_minimal = ['ADS.fAHUFlapEHAActADSInternalValuesMirror',
                          'ADS.fAHUFlapEHASetADSInternalValuesMirror',
                          'ADS.fAHUFlapETAActADSInternalValuesMirror',
                          'ADS.fAHUFlapETASetADSInternalValuesMirror',
                          'ADS.fAHUFlapODAActADSInternalValuesMirror',
                          'ADS.fAHUFlapODASetADSInternalValuesMirror',
                          'ADS.fAHUFlapSUPActADSInternalValuesMirror',
                          'ADS.fAHUFlapSUPSetADSInternalValuesMirror',
                          'ADS.fAHUFanETASpeedActADSInternalValuesMirror',
                          'ADS.fAHUFanETASpeedSetADSInternalValuesMirror',
                          'ADS.fAHUFanSUPSpeedActADSInternalValuesMirror',
                          'ADS.fAHUFanSUPSpeedSetADSInternalValuesMirror',
                          'ADS.fAHUTempEHAADSInternalValuesMirror',
                          'ADS.fAHUTempETAADSInternalValuesMirror',
                          'ADS.fAHUTempODAADSInternalValuesMirror',
                          'ADS.fAHUTempSUPADSInternalValuesMirror',
                          'ADS.fAHUPHTempAirOutADSInternalValuesMirror',
                          'ADS.fAHUCOTempOutADSInternalValuesMirror',
                          'ADS.fAHUHRBypValveAct1ADSInternalValuesMirror',
                          'ADS.fAHUHRBypValveSetADSInternalValuesMirror']
    
    datapoints_rules = ['ADS.fAHUTempEHAADSInternalValuesMirror',
                        'ADS.fAHUTempETAADSInternalValuesMirror',
                        'ADS.fAHUTempODAADSInternalValuesMirror',
                        'ADS.fAHUTempSUPADSInternalValuesMirror',
                        'ADS.fAHUPHValveActADSInternalValuesMirror',
                        'ADS.fAHUCOValveActADSInternalValuesMirror']
    
    datapoints_ahu_survey_og = ['ADS.fAHUFanSUPSpeedActADSInternalValuesMirror',
                                'ADS.fAHUTempEHAADSInternalValuesMirror',
                                'ADS.fAHUTempETAADSInternalValuesMirror',
                                'ADS.fAHUTempODAADSInternalValuesMirror',
                                'ADS.fAHUTempSUPADSInternalValuesMirror',
                                'ADS.fAHUPHValveActADSInternalValuesMirror',
                                'ADS.fAHURHValveActADSInternalValuesMirror',
                                'ADS.bAHUPHPumpOperatingADSInternalValuesMirror']
    
    datapoints_ahu_survey_ext = ['ADS.fAHUFanSUPSpeedActADSInternalValuesMirror',
                                'ADS.fAHUFanETASpeedActADSInternalValuesMirror',
                                'ADS.fAHUFanSUPSpeedSetADSInternalValuesMirror',
                                'ADS.fAHUTempETAADSInternalValuesMirror',
                                'ADS.fAHUTempODAADSInternalValuesMirror',
                                'ADS.fAHUTempSUPADSInternalValuesMirror',
                                'ADS.fAHUPHValveActADSInternalValuesMirror',
                                'ADS.fAHURHValveActADSInternalValuesMirror',
                                'ADS.fAHUCOValveActADSInternalValuesMirror',
                                'ADS.bAHUPHPumpOperatingADSInternalValuesMirror',
                                'ADS.bAHURHPumpOperatingADSInternalValuesMirror',
                                'ADS.bAHUCOPumpOperatingADSInternalValuesMirror']
                            
    
    datapoints_cross_corr = ['ADS.fAHUCOValveActADSInternalValuesMirror','ADS.fAHUCOValveSetADSInternalValuesMirror']
    datapoints_trial2_mirror = ['ADS.fAHUFlapEHAActADSInternalValuesMirror', 'ADS.fAHUFlapEHASetADSInternalValuesMirror','ADS.fAHUFlapETAActADSInternalValuesMirror','ADS.fAHUFlapETASetADSInternalValuesMirror','ADS.fAHUFanETASpeedActADSInternalValuesMirror','ADS.fAHUFanETASpeedSetADSInternalValuesMirror','ADS.fAHUFanETAVdpActADSInternalValuesMirror','ADS.fAHUFanETAVolFlowActADSInternalValuesMirror','ADS.fAHUFanSUPVolFlowActADSInternalValuesMirror','ADS.fAHUTempEHAADSInternalValuesMirror','ADS.fAHUTempETAADSInternalValuesMirror','ADS.fAHUHumEHAADSInternalValuesMirror','ADS.fAHUHumETAADSInternalValuesMirror']
    
    vals = dat.get_gps_timeseries(data=datapoints_ahu_survey_ext)
    extra = ['ADS.AHUPHTempRetPrimADS','ADS.fAHUPHTempSupADS','ADS.fAHUPHTempRetADS','ADS.fAHUHRTempOutEstimatedADS','ADS.fAHUCOTempSupPrimADS','ADS.fAHUCOTempRetPrimADS','ADS.fAHUCOTempSupADS','ADS.fAHUCOTempRetADS','ADS.fAHURHTempSupPrimADS','ADS.fAHURHTempRetPrimADS','ADS.fAHURHTempSupADS','ADS.fAHURHTempRetADS']
    vals.to_csv('data_survey_ext_15m_Nov22_Feb24.csv')
