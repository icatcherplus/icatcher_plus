import React from 'react';
import styles from './FilterBoxFrame.module.css';
import {
  Skeleton
} from '@mui/material';
import { useVideoData } from '../../state/VideoDataProvider';
import FilterGaze from './FilterGaze';
import FilterConfidence from './FilterConfidence';


function FilterBoxFrame() {

  const videoData = useVideoData();

  return (
    <div >
{/*       <h1>Filters</h1> */}
      {
        Object.keys(videoData.annotations).length !== 0 ?
          <div className={styles.box}>
            {
              Object.keys(videoData.annotations).map((key) => {
                return key==='confidence'
                  ? <FilterConfidence
                      key={key}
                      id={key}
                    />

                  : <FilterGaze
                      key={key}
                      id={key}
                    />
              })
            }
          </div>
          :
          <Skeleton
            variant="text"
//             width={playbackState.videoWidth}
            height={200}
          />
      }
    </div>
  );
}

export default FilterBoxFrame;