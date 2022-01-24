import numpy as np
def dist(x1, x2, y1, y2):
    '''
    Description: Calculate distance between 2 points
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        x1, x2, y1, y2[float]: points' coordination
    Return:
        dis[float]
    Usage:
        x1, x2, y1, y2 = 1,2,3,4
        dis = dist(x1, x2, y1, y2)
    '''
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def angle(v1, v2):
    '''
    Description: Calculate angle between 2 vectors
    Author: Yujin Wang
    Date: 2022-01-24
    Args:
        v2,v1[list]:[x1,y1,x2,y2]
    Return:
        included_angle
    Usage:
        AB = [1,-3,5,-1]
        CD = [4,1,4.5,4.5]    
        angle(AB, CD)  
    '''
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = np.arctan(dy1, dx1)
    angle1 = int(angle1 * 180/np.pi)
    # print(angle1)
    angle2 = np.arctan(dy2, dx2)
    angle2 = int(angle2 * 180/np.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle