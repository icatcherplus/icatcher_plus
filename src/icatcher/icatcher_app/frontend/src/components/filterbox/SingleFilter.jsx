import React from 'react';
import JumpButton from '../common/JumpButton';
import styles from './SingleFilter'

function SingleFilter(props) {

  const { handleJump, children } = props;

  return (
    <div className={styles.filterItem}>
      <div>{children}</div>
      <div><JumpButton handleJump={handleJump} /></div>
    </div>
  );
}

export default SingleFilter