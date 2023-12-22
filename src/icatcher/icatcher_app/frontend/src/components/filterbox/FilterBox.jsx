import React from 'react';

function FilterBox(props) {

  const { id } = props;
  const videoData = useVideoData();
  const playbackState = usePlaybackState();
  const dispatchPlaybackState = usePlaybackStateDispatch();
  const dispatchSnack = useSnacksDispatch();

  const [ colorPalette, setColorPalette ] = useState(id);
  const [ selectedLabel, setSelectedLabel ] = useState('away');
  const [ labelOptions, setLabelOptions ] = useState([])

  useEffect (()=> {
    let labels = [ ...new Set(videoData.annotations[id].slice(videoData.metadata.frameOffset))].map((l) => {return String(l)})
    setLabelOptions(labels)
    setSelectedLabel(labels[0])
    setColorPalette(id)

  }, [id, videoData.annotations])  // eslint-disable-line react-hooks/exhaustive-deps

  const computeColorArray = () => {
    let tempColorArray = []
    const palette = colorPalettes[colorPalette || 'default']
    const annotationArray = videoData.annotations[id].slice(videoData.metadata.frameOffset)
    annotationArray.forEach((a, i) => {
      tempColorArray[i] = palette[`${a}`]
    })
    return [...tempColorArray]
  }

  const jumpToNextInstance = (forward) => {
    const condition = (e) => {
      return selectedLabel === 'undefined'
        ? e === undefined
        : e === selectedLabel
    }
    let next = -1
     if(forward === true) {
      let arraySlice = videoData.annotations[id].slice(playbackState.currentFrame + 1)
      next = arraySlice.findIndex((e) => condition(e))
      if (next !== -1) { next = next + playbackState.currentFrame + 1}

    } else {
      let arraySlice = videoData.annotations[id].slice(0, playbackState.currentFrame)
      next = arraySlice.findLastIndex((e) => condition(e))
     }
     if (next !== -1) {
      dispatchPlaybackState({
        type: 'setCurrentFrame',
        currentFrame: next
       })
     }
  }


  const handleLabelChange = (e) => {
    let targetValue = e.target.value
    if(!(labelOptions.includes(targetValue))) {
      dispatchSnack(addSnack(`Value is not a valid ${id} option`, 'warning'))
      return
    }
    setSelectedLabel(targetValue)

  }

  return (
    <div className={styles.temp}>
      <AnnotationBar
        id={id}
        getColorArray={computeColorArray}
        handleJump={jumpToNextInstance}
      >
        { labelOptions.length === 0
          ? <div />
          : <Select
            className={styles.threshold}
            id={`${id}-threshold-jumper`}
            label={"Label"}
            margin="dense"
            defaultValue={selectedLabel}
            value={selectedLabel}
            onChange={(e) =>handleLabelChange(e)}
          >
            {labelOptions.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
          </Select>
        }

      </AnnotationBar>
    </div>
  );
}

export default FilterBox