/**
 * Framer Motion animation presets.
 *
 * WHY PRESETS?
 * -----------
 * Instead of writing animation props inline on every component, we define
 * them here once and spread them where needed. This keeps animations
 * consistent across the app and makes them easy to tweak globally.
 *
 * HOW FRAMER MOTION WORKS:
 * -----------------------
 * Every <motion.div> accepts three key props:
 *   - initial: the starting state (before the element is visible)
 *   - animate: the ending state (what it animates TO)
 *   - exit: the state when the element is removed (optional)
 *
 * Framer Motion interpolates between these states automatically.
 *
 * USAGE:
 * ------
 *   import { motion } from 'framer-motion'
 *   import { fadeInUp, staggerContainer } from '../utils/animations'
 *
 *   // Single element fade in
 *   <motion.div {...fadeInUp}>Hello</motion.div>
 *
 *   // Container that staggers its children
 *   <motion.div {...staggerContainer}>
 *     <motion.div {...fadeInUp}>First</motion.div>
 *     <motion.div {...fadeInUp}>Second</motion.div>
 *   </motion.div>
 */

// =============================================================================
// INDIVIDUAL ELEMENT ANIMATIONS
// =============================================================================

/** Fade in and slide up — the most common entrance animation. */
export const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.4, ease: 'easeOut' },
}

/** Fade in without movement — subtle, good for overlays and text. */
export const fadeIn = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  transition: { duration: 0.3 },
}

/** Slide in from the left — good for side panels. */
export const slideInLeft = {
  initial: { opacity: 0, x: -30 },
  animate: { opacity: 1, x: 0 },
  transition: { duration: 0.4, ease: 'easeOut' },
}

/** Slide in from the right — good for side panels. */
export const slideInRight = {
  initial: { opacity: 0, x: 30 },
  animate: { opacity: 1, x: 0 },
  transition: { duration: 0.4, ease: 'easeOut' },
}

/** Scale up from slightly smaller — good for cards and modals. */
export const scaleIn = {
  initial: { opacity: 0, scale: 0.95 },
  animate: { opacity: 1, scale: 1 },
  transition: { duration: 0.3, ease: 'easeOut' },
}

// =============================================================================
// CONTAINER ANIMATIONS (for staggering children)
// =============================================================================

/**
 * Stagger container — its children animate one after another.
 *
 * HOW STAGGERING WORKS:
 * Each child with a `variants` prop will wait for the previous child
 * to start before beginning its own animation. The `staggerChildren`
 * value (0.06s) is the delay between each child.
 *
 * USAGE:
 *   <motion.div variants={staggerContainer} initial="initial" animate="animate">
 *     <motion.div variants={staggerItem}>First</motion.div>
 *     <motion.div variants={staggerItem}>Second</motion.div>
 *   </motion.div>
 */
export const staggerContainer = {
  initial: 'initial',
  animate: 'animate',
  variants: {
    initial: {},
    animate: {
      transition: {
        staggerChildren: 0.06,
      },
    },
  },
}

/** Individual item inside a stagger container. */
export const staggerItem = {
  variants: {
    initial: { opacity: 0, y: 15 },
    animate: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.3, ease: 'easeOut' },
    },
  },
}

// =============================================================================
// PAGE TRANSITION
// =============================================================================

/**
 * Page transition preset — used in the AppShell to animate route changes.
 *
 * Combines with AnimatePresence (which detects when a component unmounts)
 * to create smooth transitions between pages.
 */
export const pageTransition = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -10 },
  transition: { duration: 0.25, ease: 'easeInOut' },
}

// =============================================================================
// HOVER & TAP (interactive feedback)
// =============================================================================

/** Subtle lift effect on hover — good for cards. */
export const hoverLift = {
  whileHover: { y: -2, transition: { duration: 0.2 } },
}

/** Scale slightly on hover — good for buttons and clickable items. */
export const hoverScale = {
  whileHover: { scale: 1.02 },
  whileTap: { scale: 0.98 },
}
