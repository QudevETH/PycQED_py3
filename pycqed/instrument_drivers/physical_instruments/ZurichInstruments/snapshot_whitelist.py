def generate_snapshot_whitelist_hdawg():
    """Returns the default snapshot white list for ZI HDAWG instruments"""
    snapshot_whitelist = {
        'IDN',
        'clockbase',
        'system_clocks_referenceclock_source',
        'system_clocks_referenceclock_status',
        'system_clocks_referenceclock_freq',
        'system_clocks_sampleclock_freq'}
    for i in range(4):
        snapshot_whitelist.update({
            f'awgs_{i}_enable',
            f'awgs_{i}_outputs_0_amplitude',
            f'awgs_{i}_outputs_1_amplitude'})
    for i in range(8):
        snapshot_whitelist.update({
            f'sigouts_{i}_direct',
            f'sigouts_{i}_offset',
            f'sigouts_{i}_on',
            f'sigouts_{i}_range',
            f'sigouts_{i}_delay'})
    return snapshot_whitelist

def generate_snapshot_whitelist_uhfqa():
    """Returns the default snapshot white list for ZI UHFQA instruments"""
    snapshot_whitelist = {
        'IDN',
        'clockbase',
        'awgs_0_enable',
        'awgs_0_outputs_0_amplitude',
        'awgs_0_outputs_1_amplitude'}
    for i in range(2):
        snapshot_whitelist.update({
            f'sigouts_{i}_offset',
            f'sigouts_{i}_on',
            f'sigouts_{i}_range',})
    return snapshot_whitelist
