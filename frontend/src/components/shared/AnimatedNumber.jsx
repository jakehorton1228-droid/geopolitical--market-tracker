/**
 * AnimatedNumber — counts up (or down) to a target value.
 *
 * WHY THIS COMPONENT?
 * -------------------
 * When FRED indicators load on the Intelligence Briefing page, we want the
 * numbers to "tick up" to their values rather than appearing instantly. This
 * draws the eye and gives a sense of the data being live.
 *
 * HOW IT WORKS:
 * -------------
 * Framer Motion's `useSpring` creates a spring-animated value that
 * smoothly interpolates from 0 to the target. `useTransform` converts
 * that animated value into a formatted string on every frame.
 *
 * USAGE:
 * ------
 *   <AnimatedNumber value={3.64} suffix="%" decimals={2} />
 *   <AnimatedNumber value={326.5} prefix="$" decimals={1} />
 *   <AnimatedNumber value={1247} decimals={0} />
 */

import { useEffect, useRef } from 'react'
import { useSpring, useTransform, motion } from 'framer-motion'

export default function AnimatedNumber({
  value,
  decimals = 1,
  prefix = '',
  suffix = '',
  duration = 1.2,
  className = '',
}) {
  // useSpring creates a smoothly animated number.
  // Think of it like a physical spring that settles at the target value.
  const spring = useSpring(0, {
    duration: duration * 1000,
    bounce: 0,
  })

  // useTransform maps the animated number to a formatted display string.
  // This runs on every animation frame (~60fps), so the number smoothly
  // updates as the spring moves.
  const display = useTransform(spring, (current) => {
    return `${prefix}${current.toFixed(decimals)}${suffix}`
  })

  // When the target value changes, update the spring target.
  // This also handles the initial mount (0 → value).
  useEffect(() => {
    spring.set(value)
  }, [spring, value])

  // motion.span renders the animated value. It automatically re-renders
  // whenever `display` changes (which is every frame during animation).
  return <motion.span className={className}>{display}</motion.span>
}
