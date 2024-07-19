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
            'awgs_{}_enable'.format(i),
            'awgs_{}_outputs_0_amplitude'.format(i),
            'awgs_{}_outputs_1_amplitude'.format(i)})
    for i in range(8):
        snapshot_whitelist.update({
            'sigouts_{}_direct'.format(i),
            'sigouts_{}_offset'.format(i),
            'sigouts_{}_on'.format(i),
            'sigouts_{}_range'.format(i),
            'sigouts_{}_delay'.format(i)})
    return snapshot_whitelist

def generate_snapshot_whitelist_uhfqa():
    """Returns the default snapshot white list for ZI UHFQA instruments"""
    snapshot_whitelist = {
        'IDN',
        'clockbase', }
    for i in range(1):
        snapshot_whitelist.update({
            'awgs_{}_enable'.format(i),
            'awgs_{}_outputs_0_amplitude'.format(i),
            'awgs_{}_outputs_1_amplitude'.format(i)})
    for i in range(2):
        snapshot_whitelist.update({
            'sigouts_{}_offset'.format(i),
            'sigouts_{}_on'.format(i),
            'sigouts_{}_range'.format(i), })
    return snapshot_whitelist
